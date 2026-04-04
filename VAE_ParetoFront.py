import numpy as np


class ParetoFront:
    """
    2D Pareto frontier for minimization in both coordinates.

    Exact mode:
        eps_x = 0, eps_y = 0

    Epsilon mode:
        Points are treated as equivalent unless they improve x or y
        by more than the specified tolerances.
    """

    def __init__(self, eps_x=0.0, eps_y=0.0):
        self.front = []
        self.eps_x = float(eps_x)
        self.eps_y = float(eps_y)

        if self.eps_x < 0 or self.eps_y < 0:
            raise ValueError("eps_x and eps_y must be nonnegative")

    def clear_front(self):
        self.front = []

    def add_points(self, x_pts, y_pts):
        x = np.asarray(x_pts, dtype=float).ravel()
        y = np.asarray(y_pts, dtype=float).ravel()

        if x.shape != y.shape:
            raise ValueError("x_pts and y_pts must have same shape")

        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]

        if x.size == 0:
            self.front = []
            return

        pts = np.column_stack([x, y])

        # Sort by x ascending, then y ascending
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        pts = pts[order]

        # Collapse points whose x values are within eps_x of each other.
        # Keep only the lowest y in each such local x-group.
        collapsed = []
        curr_x = pts[0, 0]
        curr_y = pts[0, 1]

        for i in range(1, len(pts)):
            x_i, y_i = pts[i]

            if abs(x_i - curr_x) <= self.eps_x:
                if y_i < curr_y:
                    curr_y = y_i
            else:
                collapsed.append((curr_x, curr_y))
                curr_x, curr_y = x_i, y_i

        collapsed.append((curr_x, curr_y))
        pts = np.asarray(collapsed, dtype=float)

        # Keep only points that improve the best y seen so far by > eps_y.
        front = []
        best_y = np.inf

        for x_i, y_i in pts:
            if y_i < best_y - self.eps_y:
                front.append((x_i, y_i))
                best_y = y_i

        self.front = front

    def get_front(self):
        return self.front

    def get_curve(self):
        if not self.front:
            return np.empty((0, 2), dtype=float)
        return np.asarray(self.front, dtype=float)

    def get_smooth_pareto_curve(self, num_iters=0):
        # Kept only for backward compatibility.
        return self.get_curve()