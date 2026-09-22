from collections import deque

from utils import DeltaGenerator
from utils import delt_mul, delt_div


class AdaptiveKLWeightScheduler:
    """Owns heuristic adaptive KL-weight scheduling state and update policy."""

    def __init__(self, initial_kl_weight, max_kl_weight, kl_weight_update_factor, running_window):
        self.kl_weight = float(initial_kl_weight)
        self.kl_weight_max = float(max_kl_weight)
        self.kl_weight_update_factor = float(kl_weight_update_factor)
        self.running_window = int(running_window)

        self.prev_loss_recon = float("inf")
        self.prev_loss_kl = float("inf")

        self.kl_weight_queue = deque(maxlen=self.running_window)
        self.kl_weight_queue.append(self.kl_weight)

        self._rebuild_delta_generator()

    def _rebuild_delta_generator(self):
        self.delta_gen = DeltaGenerator(delt_mul, delt_div, self.kl_weight_update_factor)
        self.inc = self.delta_gen.inc_func
        self.dec = self.delta_gen.dec_func

    def current_value(self):
        return self.kl_weight

    def current_update_factor(self):
        return self.kl_weight_update_factor

    def get_state(self):
        return {
            "kl_weight": self.kl_weight,
            "kl_weight_update_factor": self.kl_weight_update_factor,
            "prev_loss_recon": self.prev_loss_recon,
            "prev_loss_kl": self.prev_loss_kl,
            "kl_weight_queue": list(self.kl_weight_queue),
        }

    def set_state(self, state):
        self.kl_weight = float(state["kl_weight"])
        self.kl_weight_update_factor = float(state["kl_weight_update_factor"])
        self.prev_loss_recon = float(state["prev_loss_recon"])
        self.prev_loss_kl = float(state["prev_loss_kl"])
        self.kl_weight_queue.clear()
        self.kl_weight_queue.extend(float(value) for value in state["kl_weight_queue"])
        self._rebuild_delta_generator()

    def update(self, curr_loss_recon, curr_loss_kl):
        """
        Update KL weight based on current losses.

        Returns a dict of info the trainer can log/write without knowing
        the scheduler internals.
        """
        if curr_loss_recon >= self.prev_loss_recon:
            # Recon got worse: reduce KL pressure
            self.kl_weight = self.dec(self.kl_weight)
            weight_direction = "-"
        else:
            # Recon improved: increase KL pressure
            self.kl_weight = self.inc(self.kl_weight)
            weight_direction = "+"

        # Cap KL Weight
        self.kl_weight = min(self.kl_weight, self.kl_weight_max)

        self.kl_weight_queue.append(self.kl_weight)

        num_maxes = sum(1 for x in self.kl_weight_queue if x == self.kl_weight_max)
        test1 = num_maxes > 0.4 * len(self.kl_weight_queue)
        test2 = num_maxes < 0.6 * len(self.kl_weight_queue)

        update_factor_changed = False
        if test1 and test2:
            self.kl_weight_update_factor *= 0.9
            self._rebuild_delta_generator()

            self.kl_weight_queue.clear()
            self.kl_weight_queue.append(self.kl_weight)
            update_factor_changed = True

        self.prev_loss_recon = curr_loss_recon
        self.prev_loss_kl = curr_loss_kl

        return {
            "weight_direction": weight_direction,
            "kl_weight": self.kl_weight,
            "num_maxes": num_maxes,
            "test1": test1,
            "test2": test2,
            "max_kl_weight_seen": max(self.kl_weight_queue),
            "min_kl_weight_seen": min(self.kl_weight_queue),
            "window_len": len(self.kl_weight_queue),
            "update_factor": self.kl_weight_update_factor,
            "update_factor_changed": update_factor_changed,
         }
