from collections import deque

from utils import DeltaGenerator
from utils import delt_mul, delt_div


class KLController:
    """Owns adaptive KL-weight control state and update policy."""

    def __init__(self, initial_factor, max_factor, update_factor, running_window):
        self.kl_adj_factor = float(initial_factor)
        self.kl_adj_factor_max = float(max_factor)
        self.kl_adj_update_factor = float(update_factor)
        self.running_window = int(running_window)

        self.prev_loss_recon = float("inf")
        self.prev_loss_kl = float("inf")

        self.kl_adj_factor_queue = deque(maxlen=self.running_window)
        self.kl_adj_factor_queue.append(self.kl_adj_factor)

        self._rebuild_delta_generator()

    def _rebuild_delta_generator(self):
        self.delta_gen = DeltaGenerator(delt_mul, delt_div, self.kl_adj_update_factor)
        self.inc = self.delta_gen.inc_func
        self.dec = self.delta_gen.dec_func

    def current_value(self):
        return self.kl_adj_factor

    def current_update_factor(self):
        return self.kl_adj_update_factor

    def update(self, curr_loss_recon, curr_loss_kl):
        """
        Update KL adjustment factor based on current losses.

        Returns a dict of info the trainer can log/write without knowing
        the controller internals.
        """
        if curr_loss_recon >= self.prev_loss_recon:
            # Recon got worse: reduce KL pressure
            self.kl_adj_factor = self.dec(self.kl_adj_factor)
            adj_str = "-"
        else:
            # Recon improved: increase KL pressure
            self.kl_adj_factor = self.inc(self.kl_adj_factor)
            adj_str = "+"

        # Cap KL factor
        self.kl_adj_factor = min(self.kl_adj_factor, self.kl_adj_factor_max)

        self.kl_adj_factor_queue.append(self.kl_adj_factor)

        num_maxes = sum(1 for x in self.kl_adj_factor_queue if x == self.kl_adj_factor_max)
        test1 = num_maxes > 0.4 * len(self.kl_adj_factor_queue)
        test2 = num_maxes < 0.6 * len(self.kl_adj_factor_queue)

        update_factor_changed = False
        if test1 and test2:
            self.kl_adj_update_factor *= 0.9
            self._rebuild_delta_generator()

            self.kl_adj_factor_queue.clear()
            self.kl_adj_factor_queue.append(self.kl_adj_factor)
            update_factor_changed = True

        self.prev_loss_recon = curr_loss_recon
        self.prev_loss_kl = curr_loss_kl

        return {
            "adj_str": adj_str,
            "factor": self.kl_adj_factor,
            "num_maxes": num_maxes,
            "test1": test1,
            "test2": test2,
            "max_factor_seen": max(self.kl_adj_factor_queue),
            "min_factor_seen": min(self.kl_adj_factor_queue),
            "window_len": len(self.kl_adj_factor_queue),
            "update_factor": self.kl_adj_update_factor,
            "update_factor_changed": update_factor_changed,
        }