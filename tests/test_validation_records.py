"""
Tests for the held-out evaluation artifact.

val_losses_file.txt deliberately shares its first five fields with
losses_file.txt so VAE_Loss_Records parses both without modification and the
existing frontier machinery works on validation data unchanged.
"""

import pytest


VAL_HEADER = (
    "epoch -- step -- recon_loss -- kl_loss -- kl_weight  "
    "recon_sampled ssim_mu ssim_sampled active_dims kl_sum "
    "agg_post_mean agg_post_min agg_post_max n_images\n"
)


def val_line(epoch, step, recon_mu, kl, kl_weight="4.1730e+02",
             recon_sampled="61.2345"):
    return (
        f"{epoch} -- {step} -- {recon_mu} -- {kl} -- {kl_weight}  "
        f"{recon_sampled} 0.8123 0.7654 512 9.4208e+01 "
        f"1.0021e+00 8.7654e-01 1.1234e+00 12713\n"
    )


def write_val_losses(tmp_path, body, expt_name="expt_1"):
    stats = tmp_path / expt_name / "stats"
    stats.mkdir(parents=True)
    (stats / "val_losses_file.txt").write_text(VAL_HEADER + "".join(body))
    return stats / "val_losses_file.txt"


class TestValFileIsParseable:
    def test_shared_parser_reads_the_validation_file(self, tmp_path):
        from VAE_Loss_Records import read_loss_records

        path = write_val_losses(tmp_path, [
            val_line(0, 32, "1102.3311", "1.8395e-01"),
            val_line(25, 32, "402.1100", "9.1200e-02"),
            val_line(50, 32, "88.9010", "1.8400e-01"),
        ])
        records = read_loss_records(path)

        assert [r["epoch"] for r in records] == [0, 25, 50]
        assert records[1]["recon_loss"] == pytest.approx(402.11)
        assert records[1]["kl_loss"] == pytest.approx(9.12e-02)
        # recon_loss is the mu-based value, the deterministic one.
        assert records[0]["recon_loss"] == pytest.approx(1102.3311)

    def test_kl_weight_column_is_read(self, tmp_path):
        from VAE_Loss_Records import read_loss_records

        path = write_val_losses(
            tmp_path, [val_line(0, 32, "100.0", "1.0e-01", kl_weight="4.1730e+02")]
        )
        assert read_loss_records(path)[0]["kl_weight"] == pytest.approx(417.30)

    def test_frontier_machinery_accepts_validation_records(self, tmp_path):
        from VAE_Loss_Records import read_loss_records, pareto_records

        path = write_val_losses(tmp_path, [
            val_line(0, 32, "1102.3311", "1.8395e-01"),
            val_line(25, 32, "402.1100", "9.1200e-02"),
            val_line(50, 32, "500.0000", "5.0000e-01"),  # dominated
            val_line(75, 32, "88.9010", "6.0000e-01"),
        ])
        frontier = pareto_records(read_loss_records(path))

        assert [r["epoch"] for r in frontier] == [75, 25]
        assert all(
            b["kl_loss"] < a["kl_loss"]
            for a, b in zip(frontier, frontier[1:])
        )

    def test_epoch_grid_is_the_unit_of_analysis(self, tmp_path):
        from VAE_Loss_Records import read_loss_records

        # One row per validated epoch, not per training step.
        body = [val_line(e, 32, f"{1000 - e}.0", "1.0e-01")
                for e in range(0, 100, 25)]
        path = write_val_losses(tmp_path, body)
        assert len(read_loss_records(path)) == 4


class TestAggregatePosteriorMath:
    """
    Var(mu_d) + E[sigma^2_d] equals 1.0 per dimension when the aggregate
    posterior matches a standard normal prior. This checks the accumulator
    algebra used in evaluate_validation_set without needing TensorFlow.
    """

    @staticmethod
    def _agg_post(mu, sigma_sq):
        import numpy as np
        n = mu.shape[0]
        mu_mean = mu.sum(axis=0) / n
        mu_var = (mu ** 2).sum(axis=0) / n - mu_mean ** 2
        return mu_var + sigma_sq.sum(axis=0) / n

    def test_matched_prior_gives_one_per_dimension(self):
        import numpy as np
        rng = np.random.default_rng(0)
        # mu ~ N(0, 1 - s), sigma^2 = s  =>  Var(mu) + E[sigma^2] = 1
        s = 0.25
        mu = rng.normal(0.0, np.sqrt(1 - s), size=(200000, 4))
        sigma_sq = np.full_like(mu, s)
        got = self._agg_post(mu, sigma_sq)
        assert np.allclose(got, 1.0, atol=0.02)

    def test_collapsed_posterior_gives_one_as_well(self):
        import numpy as np
        # A fully collapsed dimension has mu == 0 and sigma^2 == 1.
        mu = np.zeros((1000, 3))
        sigma_sq = np.ones((1000, 3))
        assert np.allclose(self._agg_post(mu, sigma_sq), 1.0)

    def test_autoencoder_like_latent_is_far_from_one(self):
        import numpy as np
        # Low KL pressure: mu spread wide, sigma^2 tiny. This is the
        # aggregate posterior mismatch that hurts generation from the prior.
        rng = np.random.default_rng(1)
        mu = rng.normal(0.0, 4.0, size=(50000, 3))
        sigma_sq = np.full_like(mu, 1e-4)
        got = self._agg_post(mu, sigma_sq)
        assert np.all(got > 10.0)

    def test_batched_accumulation_matches_single_pass(self):
        import numpy as np
        rng = np.random.default_rng(2)
        mu = rng.normal(0.0, 1.5, size=(1000, 5))
        sigma_sq = rng.uniform(0.1, 0.9, size=(1000, 5))

        single = self._agg_post(mu, sigma_sq)

        # Uneven batches, as with drop_remainder=False.
        n = 0
        mu_sum = np.zeros(5)
        mu_sq_sum = np.zeros(5)
        sigma_sq_sum = np.zeros(5)
        for lo, hi in [(0, 400), (400, 800), (800, 1000)]:
            chunk_mu = mu[lo:hi]
            chunk_sig = sigma_sq[lo:hi]
            n += chunk_mu.shape[0]
            mu_sum += chunk_mu.sum(axis=0)
            mu_sq_sum += (chunk_mu ** 2).sum(axis=0)
            sigma_sq_sum += chunk_sig.sum(axis=0)

        mu_mean = mu_sum / n
        batched = (mu_sq_sum / n - mu_mean ** 2) + sigma_sq_sum / n
        assert np.allclose(single, batched)
