import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set random seed for reproducible layouts
np.random.seed(42)

# Simulated key vocabulary frequencies for Fake vs Real news
fake_terms = {
    'breaking': 85, 'shocking': 70, 'conspiracy': 65, 'secret': 60, 'exposed': 58,
    'unbelievable': 50, 'mainstream media': 48, 'truth': 45, 'censored': 42,
    'viral': 40, 'deep state': 38, 'bombshell': 35, 'rumor': 30, 'hoax': 28
}

real_terms = {
    'reuters': 90, 'said': 85, 'statement': 75, 'official': 70, 'government': 65,
    'spokesperson': 60, 'report': 55, 'department': 50, 'according': 48,
    'minister': 45, 'spokesman': 42, 'conference': 40, 'announced': 38, 'court': 35
}

# Generate WordCloud objects
wc_fake = WordCloud(width=600, height=400, background_color='white', 
                    colormap='Reds', max_words=50).generate_from_frequencies(fake_terms)

wc_real = WordCloud(width=600, height=400, background_color='white', 
                    colormap='Greens', max_words=50).generate_from_frequencies(real_terms)

# CREATE SUBPLOTS (1 ROW, 2 COLUMNS)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left Subplot: Fake News Vocabulary
axes[0].imshow(wc_fake, interpolation='bilinear')
axes[0].set_title('Fake News Key Vocabulary', fontsize=14, fontweight='bold', pad=12)
axes[0].axis('off')

# Right Subplot: Real News Vocabulary
axes[1].imshow(wc_real, interpolation='bilinear')
axes[1].set_title('Real News Key Vocabulary', fontsize=14, fontweight='bold', pad=12)
axes[1].axis('off')

plt.tight_layout()

# Save high-resolution PNG for paper inclusion
plt.savefig('word_cloud.png', dpi=300, bbox_inches='tight')
print("Image successfully saved as 'word_cloud.png'")