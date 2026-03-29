import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class VAESnapshotter:
    """
    Owns VAE snapshot image generation.

    This is intentionally a narrow extraction of the existing
    VAE_Trainer.snapshot_vae_behavior() logic, with minimal behavior change.
    """

    def __init__(self, raw_image_dir, latent_dim):
        self.raw_image_dir = Path(raw_image_dir)
        self.raw_image_dir.mkdir(parents=True, exist_ok=True)

        self.latent_dim = int(latent_dim)

        # Keep fixed examples / seeds so successive snapshots are comparable
        self.fixed_test_img_idxs = None
        self.fixed_gen_img_seeds = tf.random.normal(shape=[4, self.latent_dim])

    def save_snapshot(
        self,
        validation_dataset,
        vae_net,
        decoder_net,
        epoch=0,
        step=0,
        recon_loss=None,
        kl_loss=None,
    ):
        """
        Create one snapshot frame of current VAE behavior.

        Layout:
            Row 1: input images
            Row 2: reconstructions
            Row 3: decoder outputs from zero latent vectors
            Row 4: decoder outputs from random latent vectors

        First 4 columns are fixed across calls when possible.
        Last 4 columns are random each call.
        """

        # Get 1 batch from validation set and convert to numpy
        test_dataset = validation_dataset.take(1)
        output_samples = next(iter(test_dataset)).numpy()

        batch_size = output_samples.shape[0]
        if batch_size == 0:
            raise RuntimeError("Validation batch is empty; cannot create VAE snapshot")

        # VAE reconstructions
        vae_predicted, _, _ = vae_net.predict(output_samples, verbose=0)

        # 4 fixed indices + 4 random indices, clipped by batch size
        fixed_count = min(4, batch_size)
        rnd_count = min(4, batch_size)

        # Initialize fixed indices once, or reinitialize if batch size changed
        if self.fixed_test_img_idxs is None or len(self.fixed_test_img_idxs) != fixed_count:
            self.fixed_test_img_idxs = np.random.choice(
                batch_size, size=fixed_count, replace=False
            )

        # Random indices each snapshot
        rnd_test_img_idxs = np.random.choice(
            batch_size, size=rnd_count, replace=False
        )

        test_img_idxs = np.concatenate([self.fixed_test_img_idxs, rnd_test_img_idxs])

        # Decoder outputs from zero latent vectors
        avg_img_seeds = tf.zeros(shape=[len(test_img_idxs), self.latent_dim])
        avg_images = decoder_net.predict(avg_img_seeds, verbose=0)

        # Decoder outputs from fixed + random latent vectors
        rnd_gen_img_seeds = tf.random.normal(shape=[rnd_count, self.latent_dim])
        gen_img_seeds = tf.concat(
            [self.fixed_gen_img_seeds[:fixed_count], rnd_gen_img_seeds], axis=0
        )
        gen_images = decoder_net.predict(gen_img_seeds, verbose=0)

        num_idxs = len(test_img_idxs)
        fig = plt.figure(figsize=(8, 5))

        for ctr, idx in enumerate(test_img_idxs):
            # Row 1: input image
            plt.subplot(4, num_idxs, ctr + 1)
            img1 = output_samples[idx, :, :, :] * 255
            img1 = img1.astype("int32")
            plt.axis("off")
            plt.imshow(img1)

            # Row 2: reconstruction
            plt.subplot(4, num_idxs, ctr + 1 + num_idxs)
            img2 = vae_predicted[idx, :, :, :] * 255
            img2 = img2.astype("int32")
            plt.axis("off")
            plt.imshow(img2)

            # Row 3: decoder output from zero latent vector
            plt.subplot(4, num_idxs, ctr + 1 + 2 * num_idxs)
            img3 = avg_images[ctr, :, :, :] * 255
            img3 = img3.astype("int32")
            plt.axis("off")
            plt.imshow(img3)

            # Row 4: decoder output from random/fixed latent seeds
            plt.subplot(4, num_idxs, ctr + 1 + 3 * num_idxs)
            img4 = gen_images[ctr, :, :, :] * 255
            img4 = img4.astype("int32")
            plt.axis("off")
            plt.imshow(img4)

        title_str1 = f"epoch: {epoch}, step: {step}"
        if recon_loss is not None and kl_loss is not None:
            title_str2 = f"Recon Loss: {recon_loss:.4f} KL Loss: {kl_loss:.4f}"
        else:
            title_str2 = "Recon Loss: N/A KL Loss: N/A"
        fig.suptitle(title_str1 + "\n" + title_str2)

        file_name = f"image_at_epoch_{epoch:04d}_step{step:04d}.png"
        outfile = self.raw_image_dir / file_name
        plt.savefig(outfile)
        plt.close()

        logging.info(f"Saved snapshot frame to {outfile}")