from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from VAE_Anime_Datasets import Datasets


def _write_fake_images(images_dir: Path, count: int = 12) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        arr = np.full((8, 8, 3), i, dtype=np.uint8)
        png = tf.io.encode_png(arr)
        tf.io.write_file(str(images_dir / f"img_{i:03d}.png"), png)


def _dataset_batches_as_arrays(ds):
    return [batch.numpy() for batch in ds]


def test_strict_reproducibility_gives_identical_dataset_order(tmp_path):
    if not hasattr(tf, "io"):
        pytest.skip("This test requires a real TensorFlow install with tf.io")

    data_dir = tmp_path / "anime_data"
    images_dir = data_dir / "images"
    _write_fake_images(images_dir, count=12)

    common_kwargs = dict(
        output_dir=tmp_path / "out",
        seed=1234,
        data_dir=data_dir,
        strict_reproducibility=True,
    )

    ds1 = Datasets(**common_kwargs)
    ds1.set_data_params(
        batch_size=3,
        image_size=8,
        val_split=0.25,
        shuffle_buffer=32,
        train_drop_remainder=False,
    )
    ds1.data_downloaded = True
    ds1.make_train_and_validation_sets()

    ds2 = Datasets(**common_kwargs)
    ds2.set_data_params(
        batch_size=3,
        image_size=8,
        val_split=0.25,
        shuffle_buffer=32,
        train_drop_remainder=False,
    )
    ds2.data_downloaded = True
    ds2.make_train_and_validation_sets()

    train_batches_1 = _dataset_batches_as_arrays(ds1.training_dataset)
    train_batches_2 = _dataset_batches_as_arrays(ds2.training_dataset)

    val_batches_1 = _dataset_batches_as_arrays(ds1.validation_dataset)
    val_batches_2 = _dataset_batches_as_arrays(ds2.validation_dataset)

    assert len(train_batches_1) == len(train_batches_2)
    assert len(val_batches_1) == len(val_batches_2)

    for b1, b2 in zip(train_batches_1, train_batches_2):
        np.testing.assert_allclose(b1, b2)

    for b1, b2 in zip(val_batches_1, val_batches_2):
        np.testing.assert_allclose(b1, b2)


def test_strict_reproducibility_flags_are_set(tmp_path):
    if not hasattr(tf, "io"):
        pytest.skip("This test requires a real TensorFlow install with tf.io")

    data_dir = tmp_path / "anime_data"
    images_dir = data_dir / "images"
    _write_fake_images(images_dir, count=4)

    ds = Datasets(
        output_dir=tmp_path / "out",
        seed=1234,
        data_dir=data_dir,
        strict_reproducibility=True,
    )

    assert ds.num_parallel_calls == 1
    assert ds.prefetch_buffer == 1
    assert ds.reshuffle_each_iteration is True
