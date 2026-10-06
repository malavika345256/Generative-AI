"""Assignment 9: MNIST compression, reconstruction, and denoising with a VAE.

This beginner-friendly script trains one VAE for each requested latent size,
compares their reconstruction errors, and saves tables and example figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


LATENT_DIMENSIONS = (2, 16, 32)
IMAGE_VALUES = 28 * 28  # Every MNIST image contains 784 pixel values.
NUMBER_OF_EXAMPLES = 10


@keras.utils.register_keras_serializable(package="Assignment9")
class Sampling(layers.Layer):
    """Draw a latent vector using the mean and log variance from the encoder."""

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        mean, log_variance = inputs
        random_sample = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_variance) * random_sample


class VariationalAutoencoder(keras.Model):
    """A small VAE trained with reconstruction loss plus KL divergence."""

    def __init__(self, encoder: keras.Model, decoder: keras.Model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

    def call(self, images: tf.Tensor, training: bool = False) -> tf.Tensor:
        _, _, latent = self.encoder(images, training=training)
        return self.decoder(latent, training=training)

    @staticmethod
    def calculate_losses(
        targets: tf.Tensor,
        predictions: tf.Tensor,
        mean: tf.Tensor,
        log_variance: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        # Sum binary cross-entropy over each image's pixels, then average batch.
        pixel_loss = keras.losses.binary_crossentropy(targets, predictions)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(pixel_loss, axis=(1, 2)))
        kl_loss = tf.reduce_mean(
            -0.5
            * tf.reduce_sum(
                1.0 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=1
            )
        )
        return reconstruction_loss, kl_loss

    def train_step(self, data: tuple[tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
        noisy_images, clean_images = data
        with tf.GradientTape() as tape:
            mean, log_variance, latent = self.encoder(noisy_images, training=True)
            predictions = self.decoder(latent, training=True)
            reconstruction_loss, kl_loss = self.calculate_losses(
                clean_images, predictions, mean, log_variance
            )
            total_loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: tuple[tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
        noisy_images, clean_images = data
        mean, log_variance, _ = self.encoder(noisy_images, training=False)
        predictions = self.decoder(mean, training=False)
        reconstruction_loss, kl_loss = self.calculate_losses(
            clean_images, predictions, mean, log_variance
        )
        self.loss_tracker.update_state(reconstruction_loss + kl_loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}


def build_vae(latent_dim: int) -> tuple[keras.Model, keras.Model, VariationalAutoencoder]:
    """Build encoder, decoder, and their combined training model."""
    image = keras.Input(shape=(28, 28, 1), name="image")
    encoded = layers.Flatten()(image)
    encoded = layers.Dense(256, activation="relu")(encoded)
    encoded = layers.Dense(128, activation="relu")(encoded)
    mean = layers.Dense(latent_dim, name="latent_mean")(encoded)
    log_variance = layers.Dense(latent_dim, name="latent_log_variance")(encoded)
    latent = Sampling(name="latent_sample")([mean, log_variance])
    encoder = keras.Model(image, [mean, log_variance, latent], name="encoder")

    latent_input = keras.Input(shape=(latent_dim,), name="latent_code")
    decoded = layers.Dense(128, activation="relu")(latent_input)
    decoded = layers.Dense(256, activation="relu")(decoded)
    decoded = layers.Dense(IMAGE_VALUES, activation="sigmoid")(decoded)
    reconstruction = layers.Reshape((28, 28, 1))(decoded)
    decoder = keras.Model(latent_input, reconstruction, name="decoder")

    vae = VariationalAutoencoder(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    return encoder, decoder, vae


def add_gaussian_noise(
    images: np.ndarray, standard_deviation: float, rng: np.random.Generator
) -> np.ndarray:
    """Add Gaussian noise and clip pixel values back into [0, 1]."""
    noise = rng.normal(0.0, standard_deviation, size=images.shape).astype(np.float32)
    return np.clip(images + noise, 0.0, 1.0)


def unused_path(folder: Path, filename: str) -> Path:
    """Return a new path, adding a number if that filename already exists."""
    candidate = folder / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    number = 1
    while (folder / f"{stem}_{number}{suffix}").exists():
        number += 1
    return folder / f"{stem}_{number}{suffix}"


def save_image_grid(
    image_groups: list[np.ndarray], labels: list[str], output_path: Path
) -> None:
    """Save ten test examples in rows so the visual comparisons are clear."""
    figure, axes = plt.subplots(
        len(image_groups), NUMBER_OF_EXAMPLES, figsize=(15, 2.0 * len(image_groups))
    )
    for row_index, (images, label) in enumerate(zip(image_groups, labels)):
        for column_index in range(NUMBER_OF_EXAMPLES):
            axes[row_index, column_index].imshow(
                images[column_index].squeeze(), cmap="gray", vmin=0, vmax=1
            )
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(f"Image {column_index + 1}")
        axes[row_index, 0].set_ylabel(label, rotation=0, labelpad=70, va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Write a CSV result table to a path that will not overwrite old data."""
    output_path = unused_path(path.parent, path.name)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved table: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per latent size")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.35)
    parser.add_argument("--limit-train", type=int, default=0, help="0 uses all training images")
    parser.add_argument("--limit-test", type=int, default=0, help="0 uses all test images")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.noise_std < 0:
        raise ValueError("noise-std cannot be negative")

    root = Path(__file__).resolve().parent
    results_folder = root / "results"
    comparison_folder = root / "comparison_images"
    results_folder.mkdir(exist_ok=True)
    comparison_folder.mkdir(exist_ok=True)

    keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    (train_images, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images.astype(np.float32)[..., np.newaxis] / 255.0
    test_images = test_images.astype(np.float32)[..., np.newaxis] / 255.0
    if args.limit_train:
        train_images = train_images[: args.limit_train]
    if args.limit_test:
        test_images, test_labels = test_images[: args.limit_test], test_labels[: args.limit_test]
    if len(test_images) < NUMBER_OF_EXAMPLES:
        raise ValueError("At least 10 test images are required for the image comparisons")

    noisy_train = add_gaussian_noise(train_images, args.noise_std, rng)
    noisy_test = add_gaussian_noise(test_images, args.noise_std, rng)
    example_originals = test_images[:NUMBER_OF_EXAMPLES]
    example_noisy = noisy_test[:NUMBER_OF_EXAMPLES]

    summary_rows: list[list[object]] = []
    detailed_rows: list[list[object]] = []
    all_reconstructions: list[np.ndarray] = []

    for latent_dim in LATENT_DIMENSIONS:
        print(f"\nTraining VAE with latent dimension {latent_dim}...")
        encoder, decoder, vae = build_vae(latent_dim)
        vae.fit(
            noisy_train,
            train_images,
            validation_data=(noisy_test, test_images),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
        )

        # Use the encoder's mean for stable test-time reconstructions.
        test_means, _, _ = encoder.predict(test_images, batch_size=args.batch_size, verbose=0)
        noisy_means, _, _ = encoder.predict(noisy_test, batch_size=args.batch_size, verbose=0)
        reconstructions = decoder.predict(test_means, batch_size=args.batch_size, verbose=0)
        denoised = decoder.predict(noisy_means, batch_size=args.batch_size, verbose=0)
        all_reconstructions.append(reconstructions[:NUMBER_OF_EXAMPLES])

        per_image_mse = np.mean(np.square(example_originals - reconstructions[:10]), axis=(1, 2, 3))
        full_test_mse = float(np.mean(np.square(test_images - reconstructions)))
        denoising_mse = float(np.mean(np.square(example_originals - denoised[:10])))
        noisy_input_mse = float(np.mean(np.square(example_originals - example_noisy)))
        compression_ratio = IMAGE_VALUES / latent_dim

        summary_rows.append(
            [
                latent_dim,
                IMAGE_VALUES,
                f"{compression_ratio:.2f}:1",
                f"{full_test_mse:.8f}",
                f"{float(np.mean(per_image_mse)):.8f}",
                f"{noisy_input_mse:.8f}",
                f"{denoising_mse:.8f}",
            ]
        )
        for index, mse in enumerate(per_image_mse):
            detailed_rows.append(
                [latent_dim, index + 1, int(test_labels[index]), f"{float(mse):.8f}"]
            )

        # Keep the trained component models as useful, separately named files.
        for model, suffix in ((encoder, "encoder"), (decoder, "decoder")):
            model_path = unused_path(results_folder, f"vae_latent_{latent_dim}_{suffix}.keras")
            model.save(model_path)
            print(f"Saved model: {model_path}")

        save_image_grid(
            [example_originals, reconstructions[:10]],
            ["Original", "Reconstructed"],
            unused_path(comparison_folder, f"reconstruction_latent_{latent_dim}.png"),
        )
        save_image_grid(
            [example_originals, example_noisy, denoised[:10]],
            ["Original", "Noisy", "Denoised"],
            unused_path(comparison_folder, f"denoising_latent_{latent_dim}.png"),
        )

    save_image_grid(
        [example_originals, *all_reconstructions],
        ["Original", "Latent 2", "Latent 16", "Latent 32"],
        unused_path(comparison_folder, "latent_dimension_comparison.png"),
    )
    write_csv(
        results_folder / "vae_results_summary.csv",
        [
            "latent_dimension",
            "original_image_values",
            "compression_ratio",
            "full_test_reconstruction_mse",
            "10_image_reconstruction_mse_mean",
            "10_image_noisy_input_mse_mean",
            "10_image_denoised_mse_mean",
        ],
        summary_rows,
    )
    write_csv(
        results_folder / "vae_reconstruction_mse_by_image.csv",
        ["latent_dimension", "test_image_number", "digit_label", "reconstruction_mse"],
        detailed_rows,
    )

    print("\nSummary (lower MSE means closer pixel reconstruction):")
    for row in summary_rows:
        print(
            f"latent={row[0]:>2} | compression={row[2]} | "
            f"full-test MSE={row[3]} | denoised MSE={row[6]}"
        )


if __name__ == "__main__":
    main()
