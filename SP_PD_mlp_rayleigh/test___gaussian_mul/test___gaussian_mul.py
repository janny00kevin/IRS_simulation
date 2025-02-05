import torch
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for environments without a display
import matplotlib.pyplot as plt

# Generate 10,000 samples from a standard Gaussian distribution
samples1 = torch.normal(0, 1, size=(10000,))
samples2 = torch.normal(0, 1, size=(10000,))

# Plot the histogram of the samples
plt.figure(figsize=(8, 6))
bins = 100

# Define common bins to ensure consistency in the histograms
bin_edges = torch.linspace(
    min(samples1.min(), (samples1 * samples2).min()).item(),
    max(samples1.max(), (samples1 * samples2).max()).item(),
    bins + 1
).numpy()

plt.hist(samples1.numpy(), bins=bin_edges, density=True, alpha=0.7, color='blue', edgecolor='black', label='Samples1')
plt.hist((samples1 * samples2).numpy(), bins=bin_edges, density=True, alpha=0.7, color='red', edgecolor='black', label='Samples1 * Samples2')
plt.legend()

# Add labels and title
plt.title('Histogram of 10000 Samples from Standard Gaussian Distribution', fontsize=14)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Density', fontsize=12)

# Save the plot in the specified folder
save_path = "/media/commlab/TenTB/home/janny00kevin/IRS_simulation/SP_PD_mlp_rayleigh/test___gaussian_mul/std_gaussian_histogram.png"
plt.grid(alpha=0.5)
plt.savefig(save_path)
plt.close()

print(f"Histogram saved at {save_path}")

