from PIL import Image
import matplotlib.pyplot as plt

pairs = [
    ("VAE_421_B1.png",    "B1.png"),
    ("VAE_421_B10.png",   "B10.png"),
    ("VAE_421_B100.png",  "B100.png"),
    ("vae_421_B1000.png", "B1000.png"),
]
row_labels = ["B1", "B10", "B100", "B1000"]
col_titles = ["VAE_421", "Baseline"]

fig, axes = plt.subplots(4, 2, figsize=(10, 20))

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=15, fontweight="bold", pad=8)

for row, (vae_path, b_path) in enumerate(pairs):
    for col, path in enumerate([vae_path, b_path]):
        img = Image.open(path)
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(
        row_labels[row], fontsize=13, fontweight="bold",
        rotation=0, labelpad=45, va="center"
    )

plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150, bbox_inches="tight")
plt.show()