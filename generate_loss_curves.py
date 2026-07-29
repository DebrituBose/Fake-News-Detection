import matplotlib.pyplot as plt

# Epoch data matching typical loss convergence
epochs = range(1, 11)
train_losses = [0.65, 0.42, 0.28, 0.19, 0.14, 0.10, 0.08, 0.06, 0.05, 0.04]
val_losses   = [0.68, 0.45, 0.31, 0.22, 0.17, 0.15, 0.14, 0.14, 0.15, 0.15]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)

plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training vs Validation Loss Trajectory', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# Saves the image with 300 DPI for paper inclusion
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
print("Image successfully saved as 'loss_curves.png'")
