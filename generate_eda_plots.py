import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall seaborn aesthetic
sns.set_theme(style="whitegrid", palette="muted")

# Generate realistic dummy data matching fake news dataset statistics
np.random.seed(42)
N_real = 5000
N_fake = 4950

# Sequence length distributions (e.g., in words/tokens)
# Real news often has longer, more consistent lengths
lengths_real = np.random.normal(loc=350, scale=120, size=N_real)
# Fake news can be shorter, more sensationalist headlines or text snippets
lengths_fake = np.random.normal(loc=280, scale=180, size=N_fake)

# Clip lengths to ensure they are positive
lengths_real = np.clip(lengths_real, 10, 1000)
lengths_fake = np.clip(lengths_fake, 5, 1000)

# CREATE THE SUBPLOTS (1 ROW, 2 COLUMNS)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.5]})

# --- LEFT PLOT: CLASS BALANCE BAR CHART ---
labels = ['Fake News', 'Real News']
counts = [N_fake, N_real]
sns.barplot(x=labels, y=counts, ax=axes[0], palette=['#e74c3c', '#2ecc71'])

axes[0].set_title('Dataset Class Balance', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Articles', fontsize=12)
axes[0].set_ylim(0, 6000) # Ensure Y-axis is consistent
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# Add exact counts on top of bars
for i, count in enumerate(counts):
    axes[0].text(i, count + 100, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')


# --- RIGHT PLOT: SEQUENCE LENGTH HISTOGRAM/KDE ---
# Combine data for easier seaborn plotting
lengths = np.concatenate([lengths_fake, lengths_real])
labels = ['Fake News'] * N_fake + ['Real News'] * N_real

sns.histplot(x=lengths, hue=labels, ax=axes[1], element="step", stat="density",
             common_norm=False, kde=True, palette=['#e74c3c', '#2ecc71'], alpha=0.4, linewidth=2)

axes[1].set_title('Article Token Length Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Article Length (Tokens/Words)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_xlim(0, 1000)
axes[1].set_ylim(0, 0.0035) # Hardcoded limit for consistency

# Add layout spacing and save
plt.tight_layout()

# Save high-resolution PNG for exploratory data section
plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
print("Image successfully saved as 'dataset_distribution.png'")