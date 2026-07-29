import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up canvas
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Box properties
box_style = dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#2c3e50", lw=2)
highlight_style = dict(boxstyle="round,pad=0.5", fc="#e8f4f8", ec="#2980b9", lw=2)

# Define pipeline stages (Text, X position, Y position)
stages = [
    ("1. Raw News Data\n(Fake & Real Dataset)", 0.1, 0.5, box_style),
    ("2. Preprocessing\n• Tokenization\n• Stopword Removal\n• Cleaning", 0.32, 0.5, box_style),
    ("3. Embedding / Modeling\n• Token Embeddings\n• LSTM Sequential Pass\n• BERT Attention Layers", 0.58, 0.5, highlight_style),
    ("4. Classification Output\n• Sigmoid / Softmax\n• Real (0) vs. Fake (1)", 0.85, 0.5, box_style)
]

# Draw boxes
for text, x, y, style in stages:
    ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold", bbox=style)

# Draw arrows between stages
arrow_props = dict(arrowstyle="->", color="#2c3e50", lw=2.5, mutation_scale=20)

ax.annotate("", xy=(0.21, 0.5), xytext=(0.23, 0.5), arrowprops=arrow_props)
ax.annotate("", xy=(0.43, 0.5), xytext=(0.47, 0.5), arrowprops=arrow_props)
ax.annotate("", xy=(0.70, 0.5), xytext=(0.73, 0.5), arrowprops=arrow_props)

plt.title("End-to-End Fake News Detection Pipeline Architecture", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()

# Save image file
plt.savefig('pipeline_architecture.png', dpi=300, bbox_inches='tight')
print("Image successfully saved as 'pipeline_architecture.png'")