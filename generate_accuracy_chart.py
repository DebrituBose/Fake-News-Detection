import matplotlib.pyplot as plt
import seaborn as sns

# Set clean academic plotting style
sns.set_theme(style="whitegrid")

# Exact accuracy values matching Table 7 and baseline benchmarks
models = ['Naïve Bayes', 'Logistic Regression', 'Random Forest', 'LSTM Baseline', 'BERT (Proposed)']
accuracies = [83.2, 86.5, 88.9, 96.2, 98.9]  # Updated with Table 7 metrics

# Professional color palette: Light baselines -> Deep blue for BERT
colors = ['#a1dab4', '#41b6c4', '#2c7fb8', '#253494', '#081d58']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=colors, width=0.55, edgecolor='black', linewidth=0.8)

# Format Y-axis to focus on performance range
plt.ylim(75, 102)
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.xlabel('Evaluated Machine Learning & Deep Learning Models', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Benchmarks', fontsize=14, fontweight='bold', pad=15)

# Add exact percentage labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.1f}%", 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xticks(fontsize=10, rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

# Save high-resolution PNG
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("Chart successfully updated and saved as 'accuracy_comparison.png'")