"""
Tests for the shared loss-record parser.

``VAE_Loss_Records.read_loss_records`` replaced four separate parsers that
had drifted apart. The thin per-module wrappers are exercised alongside it
so a future copy-paste does not reintroduce a divergent implementation.

All of this is pure file I/O, so this module runs without TensorFlow.
"""

import pytest


HEADER = (
    "epoch -- step -- recon_loss -- kl_loss -- kl_weight  "
    "bounce_test1 bounce_test2 num_maxes max_factor min_factor window_len "
    "kl_jump_ratio max_log_var min_log_var max_grad_norm\n"
)


def loss_line(epoch, step, recon, kl, kl_weight="2.0000e-06"):
    """Reproduce ArtifactWriter.write_loss_text_line's format."""
    return (
        f"{epoch} -- {step} -- {recon} -- {kl} -- "
        f"{kl_weight}  False True 0 2.0e-06 1.0e-06 2 "
        f"1.0000e+00 -9.0000e-01 -1.0000e+00 5.0000e+00\n"
    )


def write_losses(tmp_path, body_lines, expt_name="expt_1"):
    expt_dir = tmp_path / expt_name
    stats_dir = expt_dir / "stats"
    stats_dir.mkdir(parents=True)
    (stats_dir / "losses_file.txt").write_text(HEADER + "".join(body_lines))
    return expt_dir


@pytest.fixture
def parsers():
    """Every public entry point that reads a losses_file."""
    from VAE_Loss_Records import read_loss_records as shared
    from VAE_Pareto_Comparisons import read_loss_records as comparisons
    from VAE_Pareto_Front_Intersections import read_loss_records as intersections
    from VAE_Pareto_GroupedComparisons_beta_lr import (
        read_loss_records as grouped,
    )
    from VAE_Anime_ParetoRunSummary import read_loss_records as run_summary

    return {
        "shared": lambda d: shared(d),
        "comparisons": comparisons,
        "intersections": intersections,
        "grouped": grouped,
        # This one historically took the file rather than the directory.
        "run_summary": lambda d: run_summary(d / "stats" / "losses_file.txt"),
    }


def assert_all_agree(expt_dir, parsers, expected):
    got = {name: len(fn(expt_dir)) for name, fn in parsers.items()}
    assert all(v == expected for v in got.values()), got


class TestCleanInput:
    def test_all_entry_points_agree_on_a_well_formed_file(
        self, tmp_path, parsers
    ):
        body = [
            loss_line(0, 0, "1154.8159", "1.8395e-01"),
            loss_line(0, 1, "1150.2725", "1.8477e-01"),
            loss_line(1, 0, "900.0000", "2.0000e-01"),
        ]
        expt_dir = write_losses(tmp_path, body)

        results = {
            name: [(r["epoch"], r["step"], r["recon_loss"], r["kl_loss"])
                   for r in fn(expt_dir)]
            for name, fn in parsers.items()
        }
        reference = results["shared"]
        assert len(reference) == 3
        for name, got in results.items():
            assert got == reference, f"{name} diverged from the shared parser"

    def test_header_line_is_skipped(self, tmp_path, parsers):
        expt_dir = write_losses(tmp_path, [loss_line(0, 0, "100.0", "1.0e-01")])
        assert_all_agree(expt_dir, parsers, 1)

    def test_kl_weight_is_available_to_every_caller(self, tmp_path, parsers):
        expt_dir = write_losses(
            tmp_path, [loss_line(0, 0, "100.0", "1.0e-01", kl_weight="3.5e-04")]
        )
        for name, fn in parsers.items():
            assert fn(expt_dir)[0]["kl_weight"] == pytest.approx(3.5e-04), name

    def test_line_number_records_the_source_row(self, tmp_path, parsers):
        body = [loss_line(0, i, f"{100 - i}.0", "1.0e-01") for i in range(3)]
        expt_dir = write_losses(tmp_path, body)
        for name, fn in parsers.items():
            assert [r["line_number"] for r in fn(expt_dir)] == [2, 3, 4], name

    def test_missing_file_raises(self, tmp_path, parsers):
        empty = tmp_path / "expt_missing"
        (empty / "stats").mkdir(parents=True)
        for name, fn in parsers.items():
            with pytest.raises(FileNotFoundError):
                fn(empty)

    def test_file_with_only_a_header_raises(self, tmp_path, parsers):
        expt_dir = write_losses(tmp_path, [])
        for name, fn in parsers.items():
            with pytest.raises(ValueError):
                fn(expt_dir)


class TestBadRowsAreSkippedNotFatal:
    """
    A bad row must not truncate the run. One earlier parser stopped reading
    at the first anomaly, which silently shortened every downstream curve.
    """

    def _body(self, bad_line):
        return [
            loss_line(0, 0, "100.0", "1.0e-01"),
            loss_line(0, 1, "99.0", "1.1e-01"),
            bad_line,
            loss_line(0, 3, "97.0", "1.3e-01"),
            loss_line(0, 4, "96.0", "1.4e-01"),
        ]

    @pytest.mark.parametrize(
        "bad_line",
        [
            loss_line(0, 2, "nan", "1.2e-01"),
            loss_line(0, 2, "97.5", "nan"),
            loss_line(0, 2, "inf", "1.2e-01"),
            "0 -- 2 -- not_a_number -- 1.2e-01 -- 2.0e-06\n",
            "0 -- 2 -- 98.0\n",
            loss_line(0, 2, "1e400", "1.2e-01"),
            loss_line(0, 2, "97.5", "0.0000e+00"),
            loss_line(0, 2, "97.5", "-5.0e-02"),
        ],
    )
    def test_one_bad_row_costs_exactly_one_record(
        self, bad_line, tmp_path, parsers
    ):
        expt_dir = write_losses(tmp_path, self._body(bad_line))
        assert_all_agree(expt_dir, parsers, 4)

    def test_blank_lines_are_ignored(self, tmp_path, parsers):
        body = [
            loss_line(0, 0, "100.0", "1.0e-01"),
            "\n",
            loss_line(0, 1, "99.0", "1.1e-01"),
            "   \n",
        ]
        expt_dir = write_losses(tmp_path, body)
        assert_all_agree(expt_dir, parsers, 2)


class TestValidationIsByValueNotSubstring:
    """
    Rows are judged on parsed values. Two earlier parsers searched the raw
    line for "nan"/"inf", which both missed overflowing literals and
    discarded valid rows whose trailing diagnostics contained those letters.
    """

    def test_diagnostic_field_containing_inf_does_not_discard_the_row(
        self, tmp_path, parsers
    ):
        tainted = (
            "0 -- 1 -- 99.0 -- 1.1e-01 -- 2.0e-06  "
            "False True 0 2.0e-06 inf 2 "
            "1.0000e+00 -9.0000e-01 -1.0000e+00 5.0000e+00\n"
        )
        body = [loss_line(0, 0, "100.0", "1.0e-01"), tainted]
        expt_dir = write_losses(tmp_path, body)
        assert_all_agree(expt_dir, parsers, 2)

    def test_overflowing_literal_is_caught_by_the_finiteness_check(
        self, tmp_path, parsers
    ):
        body = [
            loss_line(0, 0, "100.0", "1.0e-01"),
            loss_line(0, 1, "1e400", "1.1e-01"),
        ]
        expt_dir = write_losses(tmp_path, body)
        assert_all_agree(expt_dir, parsers, 1)

    def test_unparseable_kl_weight_leaves_the_row_usable(
        self, tmp_path, parsers
    ):
        body = [loss_line(0, 0, "100.0", "1.0e-01", kl_weight="n/a")]
        expt_dir = write_losses(tmp_path, body)
        for name, fn in parsers.items():
            records = fn(expt_dir)
            assert len(records) == 1, name
            assert records[0]["kl_weight"] is None, name


class TestPositiveKLRequirement:
    """
    log10(kl_loss) is taken downstream, so non-positive KL is rejected by
    default. The check is opt-out for callers that only need raw losses.
    """

    def test_zero_and_negative_kl_are_rejected_by_default(
        self, tmp_path, parsers
    ):
        body = [
            loss_line(0, 0, "100.0", "1.0e-01"),
            loss_line(0, 1, "99.0", "0.0000e+00"),
            loss_line(0, 2, "98.0", "-1.0e-02"),
            loss_line(0, 3, "97.0", "1.2e-01"),
        ]
        expt_dir = write_losses(tmp_path, body)
        assert_all_agree(expt_dir, parsers, 2)

    def test_every_surviving_record_is_safe_for_log10(self, tmp_path, parsers):
        from VAE_Pareto_Front_Intersections import transformed_point

        body = [
            loss_line(0, 0, "100.0", "1.0e-01"),
            loss_line(0, 1, "99.0", "0.0000e+00"),
            loss_line(0, 2, "98.0", "1.2e-01"),
        ]
        expt_dir = write_losses(tmp_path, body)
        for name, fn in parsers.items():
            for record in fn(expt_dir):
                transformed_point(record)  # must not raise

    def test_positivity_check_can_be_disabled(self, tmp_path):
        from VAE_Loss_Records import read_loss_records

        body = [
            loss_line(0, 0, "100.0", "1.0e-01"),
            loss_line(0, 1, "99.0", "0.0000e+00"),
        ]
        expt_dir = write_losses(tmp_path, body)
        assert len(read_loss_records(expt_dir, require_positive_kl=False)) == 2


class TestIterationNumbering:
    """
    "iteration" is the index among kept records everywhere. One earlier
    parser used the index over all data lines, so a skipped row shifted its
    numbering out of step with the others.
    """

    def test_numbering_is_contiguous_when_nothing_is_skipped(
        self, tmp_path, parsers
    ):
        body = [loss_line(0, i, f"{100 - i}.0", "1.0e-01") for i in range(4)]
        expt_dir = write_losses(tmp_path, body)
        for name, fn in parsers.items():
            assert [r["iteration"] for r in fn(expt_dir)] == [0, 1, 2, 3], name

    def test_numbering_stays_contiguous_across_a_skipped_row(
        self, tmp_path, parsers
    ):
        body = [
            loss_line(0, 0, "100.0", "1.0e-01"),
            loss_line(0, 1, "bad_value", "1.1e-01"),
            loss_line(0, 2, "98.0", "1.2e-01"),
        ]
        expt_dir = write_losses(tmp_path, body)
        for name, fn in parsers.items():
            records = fn(expt_dir)
            assert [r["iteration"] for r in records] == [0, 1], name
            assert [r["line_number"] for r in records] == [2, 4], name


class TestPathHandling:
    def test_shared_parser_accepts_a_directory_or_a_file(self, tmp_path):
        from VAE_Loss_Records import read_loss_records

        expt_dir = write_losses(tmp_path, [loss_line(0, 0, "100.0", "1.0e-01")])
        by_dir = read_loss_records(expt_dir)
        by_file = read_loss_records(expt_dir / "stats" / "losses_file.txt")
        assert by_dir == by_file
