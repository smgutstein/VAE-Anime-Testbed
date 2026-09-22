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
        self.fixed_validation_images = None
        # Size of the validation batch the pre-caching snapshotter sampled
        # display indices from; used only to mirror its NumPy RNG draws.
        self._legacy_batch_size = None
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

        All validation inputs are fixed across calls. For generated images,
        the first 4 latent seeds are fixed and the remaining seeds are random.
        """

        # Cache only the validation images displayed in snapshots.  Reusing this
        # small fixed batch avoids decoding a validation batch and running all
        # of it through the VAE for every snapshot.
        # NumPy RNG parity with the pre-caching snapshotter.  The VAE's
        # Sampling layer takes its op seed from np.random at trace time, and
        # the initial snapshot runs before train_step is first traced, so any
        # change in this method's NumPy consumption changes the training noise
        # stream.  The first call therefore replays the legacy footprint
        # exactly (predict() on the full batch + fixed/random index draws);
        # later calls are RNG-neutral apart from the one index draw the legacy
        # code made per snapshot.
        if self.fixed_validation_images is None:
            test_dataset = validation_dataset.take(1)
            validation_batch = next(iter(test_dataset))
            batch_size = int(tf.shape(validation_batch)[0])
            if batch_size == 0:
                raise RuntimeError("Validation batch is empty; cannot create VAE snapshot")

            snapshot_count = min(8, batch_size)
            self.fixed_validation_images = tf.identity(
                validation_batch[:snapshot_count]
            )
            self._legacy_batch_size = batch_size

            # Same call as the legacy code, so Keras traces its predict
            # function (and makes Sampling's NumPy draws) identically.
            full_predicted, _, _ = vae_net.predict(validation_batch.numpy(), verbose=0)
            vae_predicted = full_predicted[:snapshot_count]

            legacy_count = min(4, batch_size)
            np.random.choice(batch_size, size=legacy_count, replace=False)  # legacy fixed idxs
            np.random.choice(batch_size, size=legacy_count, replace=False)  # legacy random idxs
        else:
            # Eager call would draw a fresh Sampling seed from np.random;
            # the legacy (already-traced) predict path drew nothing here.
            np_state = np.random.get_state()
            vae_predicted, _, _ = vae_net(self.fixed_validation_images, training=False)
            np.random.set_state(np_state)

            legacy_count = min(4, self._legacy_batch_size)
            np.random.choice(self._legacy_batch_size, size=legacy_count, replace=False)  # legacy random idxs

        output_samples = self.fixed_validation_images

        num_idxs = int(tf.shape(output_samples)[0])
        fixed_count = min(4, num_idxs)
        rnd_count = num_idxs - fixed_count

        # Decoder outputs from zero latent vectors
        avg_img_seeds = tf.zeros(shape=[num_idxs, self.latent_dim])
        avg_images = decoder_net(avg_img_seeds, training=False)

        # Decoder outputs from fixed + random latent vectors
        rnd_gen_img_seeds = tf.random.normal(shape=[rnd_count, self.latent_dim])
        gen_img_seeds = tf.concat(
            [self.fixed_gen_img_seeds[:fixed_count], rnd_gen_img_seeds], axis=0
        )
        gen_images = decoder_net(gen_img_seeds, training=False)

        output_samples = np.asarray(output_samples)
        vae_predicted = np.asarray(vae_predicted)
        avg_images = np.asarray(avg_images)
        gen_images = np.asarray(gen_images)

        fig = plt.figure(figsize=(8, 5))

        for ctr in range(num_idxs):
            # Row 1: input image
            plt.subplot(4, num_idxs, ctr + 1)
            img1 = output_samples[ctr, :, :, :] * 255
            img1 = img1.astype("int32")
            plt.axis("off")
            plt.imshow(img1)

            # Row 2: reconstruction
            plt.subplot(4, num_idxs, ctr + 1 + num_idxs)
            img2 = vae_predicted[ctr, :, :, :] * 255
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
