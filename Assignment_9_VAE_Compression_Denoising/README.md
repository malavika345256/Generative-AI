# Assignment 9: VAE Compression and Denoising

This project trains Variational Autoencoders on MNIST with latent dimensions 2, 16, and 32. The script compares compression ratios, reconstructions, and Gaussian-noise denoising results.

## Run

From this folder, run:

```powershell
python .\assignment9_vae.py
```

The script uses TensorFlow/Keras. It downloads MNIST on the first run if the dataset is not cached. By default, each latent size is trained for five epochs on the full MNIST training set. Optional arguments include `--epochs`, `--batch-size`, `--noise-std`, `--limit-train`, and `--limit-test`; use `--help` to see their descriptions.

## Results

- `results/vae_results_summary.csv` contains compression ratios and reconstruction/denoising MSE summary values.
- `results/vae_reconstruction_mse_by_image.csv` contains the reconstruction MSE for each of the ten selected test images and each latent size.
- `results/vae_latent_*_encoder.keras` and `results/vae_latent_*_decoder.keras` are the trained model components.
- `comparison_images/` contains reconstruction and denoising grids for each latent size and a combined latent-size comparison.

Compression ratios compare 784 float32 input pixel values with float32 latent means; they do not include model weights or file metadata.
