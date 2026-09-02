"""
Tests for the 2D geometry used to locate Pareto curve intersections.

These functions decide which saved snapshot frames are compared in the
intersection-based figures, so an error here selects the wrong images
without raising anything.

All functions under test are pure, so this module runs without TensorFlow.
"""

import math

import pytest


class TestVectorHelpers:
    @pytest.fixture(autouse=True)
    def _import(self):
        import VAE_Pareto_Front_Intersections as mod
        self.mod = mod

    def test_cross_is_the_2d_scalar_cross_product(self):
        assert self.mod.cross((1.0, 0.0), (0.0, 1.0)) == 1.0
        assert self.mod.cross((0.0, 1.0), (1.0, 0.0)) == -1.0

    def test_cross_of_parallel_vectors_is_zero(self):
        assert self.mod.cross((2.0, 4.0), (1.0, 2.0)) == 0.0

    def test_subtract_add_are_inverses(self):
        a = (3.0, -7.0)
        b = (1.5, 2.5)
        assert self.mod.add(self.mod.subtract(a, b), b) == a

    def test_multiply_scales_both_components(self):
        assert self.mod.multiply((2.0, -3.0), 2.5) == (5.0, -7.5)

    def test_dot_product(self):
        assert self.mod.dot((3.0, 4.0), (2.0, 1.0)) == 10.0
        assert self.mod.dot((1.0, 0.0), (0.0, 1.0)) == 0.0

    def test_distance_sq_is_squared_euclidean(self):
        assert self.mod.distance_sq((0.0, 0.0), (3.0, 4.0)) == 25.0
        assert self.mod.distance_sq((1.0, 1.0), (1.0, 1.0)) == 0.0

    def test_midpoint(self):
        assert self.mod.midpoint((0.0, 0.0), (4.0, 6.0)) == (2.0, 3.0)
        assert self.mod.midpoint((-2.0, 5.0), (2.0, -5.0)) == (0.0, 0.0)


class TestSegmentIntersection:
    @pytest.fixture(autouse=True)
    def _import(self):
        from VAE_Pareto_Front_Intersections import segment_intersection
        self.f = segment_intersection

    def test_clean_crossing(self):
        point = self.f((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0))
        assert point == pytest.approx((1.0, 1.0))

    def test_crossing_is_symmetric_in_argument_order(self):
        a = self.f((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0))
        b = self.f((0.0, 2.0), (2.0, 0.0), (0.0, 0.0), (2.0, 2.0))
        assert a == pytest.approx(b)

    def test_parallel_segments_return_none(self):
        assert self.f((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)) is None

    def test_collinear_overlapping_segments_return_none(self):
        # Documented behavior: only non-collinear segments are handled, so
        # an overlapping pair reports no intersection rather than a range.
        assert self.f((0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (3.0, 0.0)) is None

    def test_segments_touching_at_an_endpoint(self):
        point = self.f((0.0, 0.0), (1.0, 1.0), (1.0, 1.0), (2.0, 0.0))
        assert point == pytest.approx((1.0, 1.0))

    def test_t_junction(self):
        # Second segment ends on the interior of the first.
        point = self.f((0.0, 0.0), (4.0, 0.0), (2.0, 3.0), (2.0, 0.0))
        assert point == pytest.approx((2.0, 0.0))

    def test_infinite_lines_cross_but_segments_do_not(self):
        # The lines y=x and y=-x+10 meet at (5,5), outside both segments.
        assert self.f((0.0, 0.0), (1.0, 1.0), (8.0, 2.0), (9.0, 1.0)) is None

    def test_disjoint_segments_return_none(self):
        assert self.f((0.0, 0.0), (1.0, 0.0), (5.0, 5.0), (6.0, 6.0)) is None

    def test_returned_point_lies_on_both_segments(self):
        p, p2 = (0.0, 1.0), (5.0, 4.0)
        q, q2 = (1.0, 5.0), (4.0, 0.0)
        point = self.f(p, p2, q, q2)
        assert point is not None

        def on_segment(pt, a, b):
            cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
            within = (
                min(a[0], b[0]) - 1e-9 <= pt[0] <= max(a[0], b[0]) + 1e-9
                and min(a[1], b[1]) - 1e-9 <= pt[1] <= max(a[1], b[1]) + 1e-9
            )
            return abs(cross) < 1e-9 and within

        assert on_segment(point, p, p2)
        assert on_segment(point, q, q2)

    def test_degenerate_zero_length_segment_returns_none(self):
        assert self.f((1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (2.0, 2.0)) is None


class TestClosestPointOnSegment:
    @pytest.fixture(autouse=True)
    def _import(self):
        from VAE_Pareto_Front_Intersections import closest_point_on_segment
        self.f = closest_point_on_segment

    def test_projection_falls_inside_the_segment(self):
        assert self.f((2.0, 5.0), (0.0, 0.0), (4.0, 0.0)) == pytest.approx((2.0, 0.0))

    def test_projection_clamps_past_the_start(self):
        assert self.f((-3.0, 1.0), (0.0, 0.0), (4.0, 0.0)) == pytest.approx((0.0, 0.0))

    def test_projection_clamps_past_the_end(self):
        assert self.f((9.0, 1.0), (0.0, 0.0), (4.0, 0.0)) == pytest.approx((4.0, 0.0))

    def test_point_already_on_segment_maps_to_itself(self):
        assert self.f((1.5, 0.0), (0.0, 0.0), (4.0, 0.0)) == pytest.approx((1.5, 0.0))

    def test_degenerate_segment_returns_its_only_endpoint(self):
        assert self.f((5.0, 5.0), (1.0, 2.0), (1.0, 2.0)) == (1.0, 2.0)

    def test_result_is_never_farther_than_either_endpoint(self):
        point = (3.0, 7.0)
        a, b = (0.0, 0.0), (10.0, 2.0)
        got = self.f(point, a, b)

        def d2(u, v):
            return (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2

        assert d2(point, got) <= d2(point, a) + 1e-12
        assert d2(point, got) <= d2(point, b) + 1e-12


class TestClosestPointsBetweenSegments:
    @pytest.fixture(autouse=True)
    def _import(self):
        from VAE_Pareto_Front_Intersections import closest_points_between_segments
        self.f = closest_points_between_segments

    def test_intersecting_segments_give_zero_distance(self):
        pa, pb, dist_sq = self.f(
            (0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)
        )
        assert dist_sq == 0.0
        assert pa == pytest.approx((1.0, 1.0))
        assert pa == pb

    def test_parallel_segments(self):
        pa, pb, dist_sq = self.f(
            (0.0, 0.0), (4.0, 0.0), (0.0, 3.0), (4.0, 3.0)
        )
        assert dist_sq == pytest.approx(9.0)
        assert pa[1] == pytest.approx(0.0)
        assert pb[1] == pytest.approx(3.0)

    def test_disjoint_segments_closest_at_endpoints(self):
        pa, pb, dist_sq = self.f(
            (0.0, 0.0), (1.0, 0.0), (4.0, 0.0), (5.0, 0.0)
        )
        assert dist_sq == pytest.approx(9.0)
        assert pa == pytest.approx((1.0, 0.0))
        assert pb == pytest.approx((4.0, 0.0))

    def test_matches_dense_sampling_reference(self):
        # For non-intersecting 2D segments the closest pair always involves
        # an endpoint, but a dense scan is an independent check.
        p1, p2 = (0.0, 0.0), (5.0, 1.0)
        q1, q2 = (1.0, 4.0), (6.0, 2.0)

        _, _, dist_sq = self.f(p1, p2, q1, q2)

        n = 400
        best = float("inf")
        for i in range(n + 1):
            t = i / n
            ax = p1[0] + (p2[0] - p1[0]) * t
            ay = p1[1] + (p2[1] - p1[1]) * t
            for j in range(n + 1):
                u = j / n
                bx = q1[0] + (q2[0] - q1[0]) * u
                by = q1[1] + (q2[1] - q1[1]) * u
                d = (ax - bx) ** 2 + (ay - by) ** 2
                if d < best:
                    best = d

        assert dist_sq == pytest.approx(best, rel=1e-3)


class TestExactCurveIntersections:
    @pytest.fixture(autouse=True)
    def _import(self):
        from VAE_Pareto_Front_Intersections import exact_curve_intersections
        self.f = exact_curve_intersections

    def test_single_crossing(self):
        a = [(0.0, 0.0), (4.0, 4.0)]
        b = [(0.0, 4.0), (4.0, 0.0)]
        hits = self.f(a, b)
        assert len(hits) == 1
        assert hits[0]["point"] == pytest.approx((2.0, 2.0))
        assert hits[0]["segment_a"] == 0
        assert hits[0]["segment_b"] == 0

    def test_multiple_crossings_are_all_reported(self):
        a = [(0.0, 0.0), (2.0, 2.0), (4.0, 0.0)]
        b = [(0.0, 1.0), (4.0, 1.0)]
        hits = self.f(a, b)
        assert len(hits) == 2
        xs = sorted(h["point"][0] for h in hits)
        assert xs == pytest.approx([1.0, 3.0])

    def test_no_crossing_returns_empty_list(self):
        a = [(0.0, 0.0), (4.0, 0.0)]
        b = [(0.0, 9.0), (4.0, 9.0)]
        assert self.f(a, b) == []

    def test_shared_vertex_is_not_double_counted(self):
        # Both curves have a vertex at (2,2). Two distinct segment pairs
        # (a0,b0) and (a1,b1) each report it, so the duplicate filter has
        # to collapse them. No pair here is collinear, which would
        # short-circuit before the filter runs.
        a = [(0.0, 0.0), (2.0, 2.0), (4.0, 0.0)]
        b = [(0.0, 4.0), (2.0, 2.0), (4.0, 4.0)]
        hits = self.f(a, b)
        assert len(hits) == 1
        assert hits[0]["point"] == pytest.approx((2.0, 2.0))
        # The first reporting pair is the one retained.
        assert (hits[0]["segment_a"], hits[0]["segment_b"]) == (0, 0)

    def test_segment_indices_point_at_the_right_pieces(self):
        a = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        b = [(2.5, -1.0), (2.5, 1.0)]
        hits = self.f(a, b)
        assert len(hits) == 1
        assert hits[0]["segment_a"] == 2
        assert hits[0]["segment_b"] == 0

    def test_single_point_curve_has_no_segments(self):
        assert self.f([(1.0, 1.0)], [(0.0, 0.0), (2.0, 2.0)]) == []


class TestClosestCurveApproach:
    @pytest.fixture(autouse=True)
    def _import(self):
        from VAE_Pareto_Front_Intersections import closest_curve_approach
        self.f = closest_curve_approach

    def test_intersecting_curves_report_zero_distance(self):
        a = [(0.0, 0.0), (4.0, 4.0)]
        b = [(0.0, 4.0), (4.0, 0.0)]
        best = self.f(a, b)
        assert best["distance_sq"] == 0.0
        assert best["point_a"] == pytest.approx((2.0, 2.0))

    def test_finds_the_globally_closest_segment_pair(self):
        a = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        b = [(0.0, 10.0), (1.0, 0.5), (2.0, 10.0)]
        best = self.f(a, b)
        assert best["distance_sq"] == pytest.approx(0.25)
        assert best["point_a"] == pytest.approx((1.0, 0.0))
        assert best["point_b"] == pytest.approx((1.0, 0.5))

    def test_returns_none_when_a_curve_has_no_segments(self):
        assert self.f([(0.0, 0.0)], [(1.0, 1.0), (2.0, 2.0)]) is None

    def test_agrees_with_brute_force_over_all_segment_pairs(self):
        from VAE_Pareto_Front_Intersections import closest_points_between_segments

        a = [(0.0, 0.0), (2.0, 3.0), (5.0, 1.0), (7.0, 4.0)]
        b = [(0.0, 6.0), (3.0, 5.0), (6.0, 7.0)]

        best = self.f(a, b)

        brute = min(
            closest_points_between_segments(a[i], a[i + 1], b[j], b[j + 1])[2]
            for i in range(len(a) - 1)
            for j in range(len(b) - 1)
        )
        assert best["distance_sq"] == pytest.approx(brute)


class TestTransformAndNormalize:
    @pytest.fixture(autouse=True)
    def _import(self):
        import VAE_Pareto_Front_Intersections as mod
        self.mod = mod

    def test_transformed_point_takes_log10_of_kl_only(self):
        got = self.mod.transformed_point({"recon_loss": 50.0, "kl_loss": 0.01})
        assert got == pytest.approx((50.0, -2.0))

    def test_normalizer_maps_joint_extremes_to_unit_square(self):
        a = [{"recon_loss": 10.0, "kl_loss": 0.1}]
        b = [{"recon_loss": 50.0, "kl_loss": 10.0}]

        normalize, _ = self.mod.make_normalizer(a, b)
        assert normalize(10.0, -1.0) == pytest.approx((0.0, 0.0))
        assert normalize(50.0, 1.0) == pytest.approx((1.0, 1.0))

    def test_denormalize_inverts_normalize(self):
        a = [{"recon_loss": 30.0, "kl_loss": 0.5}]
        b = [{"recon_loss": 90.0, "kl_loss": 4.0}]

        normalize, denormalize = self.mod.make_normalizer(a, b)
        for x, y in [(30.0, -0.3), (60.0, 0.2), (90.0, 0.6)]:
            assert denormalize(*normalize(x, y)) == pytest.approx((x, y))

    def test_degenerate_axis_falls_back_to_unit_scale(self):
        # Identical recon_loss on every record leaves x_scale at zero.
        a = [{"recon_loss": 7.0, "kl_loss": 0.1}]
        b = [{"recon_loss": 7.0, "kl_loss": 1.0}]

        normalize, _ = self.mod.make_normalizer(a, b)
        assert normalize(7.0, -1.0) == pytest.approx((0.0, 0.0))
        assert normalize(9.0, -1.0) == pytest.approx((2.0, 0.0))

    def test_normalized_curve_applies_both_steps(self):
        frontier = [
            {"recon_loss": 10.0, "kl_loss": 0.1},
            {"recon_loss": 50.0, "kl_loss": 10.0},
        ]
        normalize, _ = self.mod.make_normalizer(frontier, [])
        curve = self.mod.normalized_curve(frontier, normalize)
        assert curve[0] == pytest.approx((0.0, 0.0))
        assert curve[1] == pytest.approx((1.0, 1.0))

    def test_normalization_preserves_intersection_topology(self):
        # Normalization is affine per axis, so a crossing stays a crossing.
        a = [{"recon_loss": 10.0, "kl_loss": 1.0},
             {"recon_loss": 50.0, "kl_loss": 0.01}]
        b = [{"recon_loss": 10.0, "kl_loss": 0.01},
             {"recon_loss": 50.0, "kl_loss": 1.0}]

        raw_a = [self.mod.transformed_point(r) for r in a]
        raw_b = [self.mod.transformed_point(r) for r in b]
        raw_hits = self.mod.exact_curve_intersections(raw_a, raw_b)

        normalize, _ = self.mod.make_normalizer(a, b)
        norm_hits = self.mod.exact_curve_intersections(
            self.mod.normalized_curve(a, normalize),
            self.mod.normalized_curve(b, normalize),
        )
        assert len(raw_hits) == len(norm_hits) == 1
