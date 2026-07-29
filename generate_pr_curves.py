import numpy as np
import matplotlib.pyplot as plt

# Generate smooth Precision-Recall curves
recall = np.linspace(0.0, 1.0, 100)

# Simulate Precision values corresponding to Recall levels
# BERT: Extremely high precision even at high recall levels (AP = 0.99)
precision_bert = 1.0 - 0.02 * (recall ** 3)

# LSTM: Tighter decay as recall increases (AP = 0.96)
precision_lstm = 1.0 - 0.08 * (recall ** 2)

# Baseline / Logistic Regression (AP = 0.87)
precision_lr = 1.0 - 0.25 * (recall ** 1.5)

plt.figure(figsize=(8, 6))

# Plot Precision-Recall curves
plt.plot(recall, precision_bert, label='BERT (AP = 0.99)', color='#1f77b4', linewidth=2.5)
plt.plot(recall, precision_lstm, label='LSTM (AP = 0.96)', color='#ff7f0e', linewidth=2.5)
plt.plot(recall, precision_lr, label='Logistic Regression (AP = 0.87)', color='#2ca02c', linestyle='--', linewidth=2)

# Graph Formatting
plt.xlim([0.0, 1.02])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
plt.title('Precision-Recall Curve Comparison across Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# Save high-resolution image
plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
print("Image successfully saved as 'precision_recall_curves.png'")