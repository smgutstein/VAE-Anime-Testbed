"""
Tests for the Pareto frontier primitives.

Two concerns are covered here:

1. ``ParetoFront`` itself -- the exact and epsilon-tolerant frontier used to
   build ParetoPoints.txt.
2. Agreement between the five separate frontier implementations in the repo.
   ``ParetoFront.add_points``, the three ``pareto_records`` copies, and
   ``ParetoRunSummary.find_pareto_records`` are independent code paths that
   are documented as computing the same thing, so they are pinned against
   each other here.
"""

import math
import random

import numpy as np
import pytest


def brute_force_front(points):
    """
    O(n^2) reference frontier for minimization in both coordinates.

    A point is kept when no other point is <= it in both coordinates and
    strictly better in at least one. Ties on x are resolved by keeping only
    the lowest y, matching ParetoFront's collapse step.
    """
    keep = []
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if xj <= xi and yj <= yi and (xj < xi or yj < yi):
                dominated = True
                break
        if not dominated:
            keep.append((xi, yi))

    # Collapse duplicate x values, keeping the lowest y.
    best_by_x = {}
    for x, y in keep:
        if x not in best_by_x or y < best_by_x[x]:
            best_by_x[x] = y
    return sorted(best_by_x.items())


class TestParetoFrontExact:
    @pytest.fixture(autouse=True)
    def _import(self):
        from VAE_ParetoFront import ParetoFront
        self.ParetoFront = ParetoFront

    def test_known_input_gives_known_front(self):
        pf = self.ParetoFront()
        # (2,5) is dominated by (1,4); (4,6) is dominated by (3,2).
        pf.add_points([1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 2.0, 6.0])
        assert pf.get_front() == [(1.0, 4.0), (3.0, 2.0)]

    def test_front_has_increasing_x_and_decreasing_y(self):
        rng = random.Random(0)
        xs = [rng.uniform(0, 100) for _ in range(200)]
        ys = [rng.uniform(0, 100) for _ in range(200)]

        pf = self.ParetoFront()
        pf.add_points(xs, ys)
        front = pf.get_front()

        assert len(front) >= 1
        for (x0, y0), (x1, y1) in zip(front, front[1:]):
            assert x1 > x0
            assert y1 < y0

    def test_no_front_point_dominates_another(self):
        rng = random.Random(1)
        xs = [rng.uniform(0, 10) for _ in range(150)]
        ys = [rng.uniform(0, 10) for _ in range(150)]

        pf = self.ParetoFront()
        pf.add_points(xs, ys)
        front = pf.get_front()

        for i, (xi, yi) in enumerate(front):
            for j, (xj, yj) in enumerate(front):
                if i == j:
                    continue
                assert not (xj <= xi and yj <= yi)

    def test_matches_brute_force_reference(self):
        rng = random.Random(2)
        for _ in range(20):
            pts = [
                (round(rng.uniform(0, 20), 3), round(rng.uniform(0, 20), 3))
                for _ in range(60)
            ]
            pf = self.ParetoFront()
            pf.add_points([p[0] for p in pts], [p[1] for p in pts])
            assert pf.get_front() == brute_force_front(pts)

    def test_duplicate_x_keeps_lowest_y(self):
        pf = self.ParetoFront()
        pf.add_points([5.0, 5.0, 5.0], [3.0, 1.0, 7.0])
        assert pf.get_front() == [(5.0, 1.0)]

    def test_identical_points_collapse_to_one(self):
        pf = self.ParetoFront()
        pf.add_points([2.0] * 10, [8.0] * 10)
        assert pf.get_front() == [(2.0, 8.0)]

    def test_single_point(self):
        pf = self.ParetoFront()
        pf.add_points([1.5], [2.5])
        assert pf.get_front() == [(1.5, 2.5)]

    def test_empty_input_gives_empty_front(self):
        pf = self.ParetoFront()
        pf.add_points([], [])
        assert pf.get_front() == []
        assert pf.get_curve().shape == (0, 2)

    def test_non_finite_values_are_dropped(self):
        pf = self.ParetoFront()
        pf.add_points(
            [1.0, float("nan"), 2.0, float("inf"), 3.0],
            [9.0, 0.0, 5.0, 0.0, float("nan")],
        )
        # Only (1,9) and (2,5) survive filtering; (2,5) improves on y.
        assert pf.get_front() == [(1.0, 9.0), (2.0, 5.0)]

    def test_all_non_finite_gives_empty_front(self):
        pf = self.ParetoFront()
        pf.add_points([float("nan"), float("inf")], [float("nan"), 1.0])
        assert pf.get_front() == []

    def test_get_curve_returns_front_as_array(self):
        pf = self.ParetoFront()
        pf.add_points([1.0, 3.0], [4.0, 2.0])
        curve = pf.get_curve()
        assert isinstance(curve, np.ndarray)
        assert curve.shape == (2, 2)
        np.testing.assert_allclose(curve, [[1.0, 4.0], [3.0, 2.0]])

    def test_mismatched_input_lengths_raise(self):
        pf = self.ParetoFront()
        with pytest.raises(ValueError):
            pf.add_points([1.0, 2.0], [1.0])

    def test_negative_epsilon_rejected(self):
        with pytest.raises(ValueError):
            self.ParetoFront(eps_x=-1.0)
        with pytest.raises(ValueError):
            self.ParetoFront(eps_y=-1.0)

    def test_clear_front_empties_it(self):
        pf = self.ParetoFront()
        pf.add_points([1.0, 3.0], [4.0, 2.0])
        pf.clear_front()
        assert pf.get_front() == []


class TestParetoFrontDocumentedQuirks:
    """
    These pin behavior that is surprising from the method names. They are
    not assertions that the behavior is desirable, only that it is current
    and should not change silently.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from VAE_ParetoFront import ParetoFront
        self.ParetoFront = ParetoFront

    def test_add_points_replaces_rather_than_accumulates(self):
        pf = self.ParetoFront()
        pf.add_points([1.0], [1.0])
        pf.add_points([10.0], [10.0])
        # Despite the name, the second call discards the first batch.
        assert pf.get_front() == [(10.0, 10.0)]

    def test_eps_x_groups_are_anchored_not_chained(self):
        # Grouping is measured against the group's first point, not the
        # previous point. Here 2.4 is 0.9 from its predecessor (within
        # eps_x) but 1.4 from the anchor at 1.0 (outside it), so it starts
        # a new group. A rewrite that compared against the previous point
        # would merge all three.
        #
        # Note: values are kept well away from exactly eps_x apart. The
        # comparison is a bare float <=, so points separated by exactly
        # eps_x fall on either side depending on representation error.
        pf = self.ParetoFront(eps_x=1.0)
        pf.add_points([1.0, 1.5, 2.4], [4.0, 3.0, 2.0])
        assert pf.get_front() == [(1.0, 3.0), (2.4, 2.0)]

    def test_eps_y_requires_strict_improvement_beyond_tolerance(self):
        pf = self.ParetoFront(eps_y=1.0)
        # Second point improves y by exactly 1.0, which is not > eps_y.
        pf.add_points([1.0, 2.0], [5.0, 4.0])
        assert pf.get_front() == [(1.0, 5.0)]

        pf2 = self.ParetoFront(eps_y=1.0)
        pf2.add_points([1.0, 2.0], [5.0, 3.5])
        assert len(pf2.get_front()) == 2


class TestFrontierImplementationsAgree:
    """
    Five independent frontier implementations exist:

        VAE_ParetoFront.ParetoFront.add_points
        VAE_Pareto_Comparisons.pareto_records
        VAE_Pareto_Front_Intersections.pareto_records
        VAE_Pareto_GroupedComparisons_beta_lr.pareto_records
        VAE_Anime_ParetoRunSummary.find_pareto_records

    All four function copies carry docstrings saying they mirror
    ``ParetoFront``. This pins that claim.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from VAE_ParetoFront import ParetoFront
        from VAE_Pareto_Comparisons import pareto_records as pr_comparisons
        from VAE_Pareto_Front_Intersections import pareto_records as pr_intersections
        from VAE_Pareto_GroupedComparisons_beta_lr import (
            pareto_records as pr_grouped,
        )
        from VAE_Anime_ParetoRunSummary import find_pareto_records as pr_summary
        from VAE_Loss_Records import pareto_records as pr_shared

        self.ParetoFront = ParetoFront
        self.record_impls = {
            "comparisons": pr_comparisons,
            "intersections": pr_intersections,
            "grouped": pr_grouped,
            "run_summary": pr_summary,
            "shared": pr_shared,
        }

    @staticmethod
    def _records(points):
        # find_pareto_records also sorts on "iteration", so it is supplied
        # here. The other three implementations ignore it.
        return [
            {"recon_loss": x, "kl_loss": y, "iteration": i}
            for i, (x, y) in enumerate(points)
        ]

    @staticmethod
    def _as_pairs(records):
        return [(r["recon_loss"], r["kl_loss"]) for r in records]

    def _random_points(self, seed, n=80):
        rng = random.Random(seed)
        return [
            (round(rng.uniform(0, 50), 4), round(rng.uniform(0, 5), 4))
            for _ in range(n)
        ]

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_all_four_record_impls_agree(self, seed):
        points = self._random_points(seed)
        records = self._records(points)

        results = {
            name: self._as_pairs(fn(records))
            for name, fn in self.record_impls.items()
        }

        reference = results["comparisons"]
        for name, got in results.items():
            assert got == reference, f"{name} diverged from comparisons"

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_record_impls_agree_with_pareto_front_class(self, seed):
        points = self._random_points(seed)

        pf = self.ParetoFront()
        pf.add_points([p[0] for p in points], [p[1] for p in points])
        expected = [tuple(map(float, p)) for p in pf.get_front()]

        got = self._as_pairs(self.record_impls["comparisons"](self._records(points)))
        assert got == expected

    def test_duplicate_recon_loss_handled_identically(self):
        points = [(1.0, 5.0), (1.0, 2.0), (1.0, 9.0), (3.0, 1.0)]
        records = self._records(points)

        results = [self._as_pairs(fn(records)) for fn in self.record_impls.values()]
        assert all(r == results[0] for r in results)
        assert results[0] == [(1.0, 2.0), (3.0, 1.0)]

    def test_short_frontiers_are_returned_not_rejected(self):
        """
        All entry points return a short frontier. The two-point requirement
        is enforced by retained_pareto_records, closer to where it matters.
        """
        records = self._records([(1.0, 1.0), (2.0, 5.0), (3.0, 9.0)])

        assert len(self.record_impls["comparisons"](records)) == 1
        assert len(self.record_impls["grouped"](records)) == 1
        assert len(self.record_impls["run_summary"](records)) == 1
        assert len(self.record_impls["shared"](records)) == 1
        # The short-frontier guard now lives in retained_pareto_records,
        # not in the frontier function itself.
        assert len(self.record_impls["intersections"](records)) == 1
