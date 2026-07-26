import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import Standard Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# --- 1. LOAD & PREPARE DATASET ---
print("Loading dataset...")
# Load sample dataset (replace with your dataset load if needed)
data = {
    'text': [
        "The prime minister announced a new economic policy today.",
        "Scientists confirmed the discovery of a new planet orbiting a star.",
        "Tech conglomerates unveiled an energy efficient AI chip.",
        "Archaeologists unearthed an ancient settlement in Egypt.",
        "World leaders pledged billions to global ocean cleanup.",
        "SHOCKING SECRET: Alien spacecraft hidden under ocean!",
        "Miracle ancient root cures all modern diseases in 24 hours!",
        "ALERT: New wireless 6G towers cause instant memory loss!",
        "BREAKING: Secret underground bunker found containing stolen gold!",
        "Hidden documents prove history books were rewritten in 1950."
    ] * 50, # Duplicated to create sample dataset batch
    'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 50
}
df = pd.DataFrame(data)

# Split into Train / Test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 2. DEFINE ALGORITHMS TO COMPARE ---
models = {
    "Logistic Regression": LogisticRegression(),
    "Naïve Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# --- 3. RUN EVALUATION LOOP ---
results = []

print("\nRunning Models & Extracting Metrics...")
for name, model in models.items():
    # Train
    model.fit(X_train_vec, y_train)
    # Predict
    y_pred = model.predict(X_test_vec)
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

print("\n================ COMPARISON TABLE ================")
print(results_df.to_string(index=False))
print("==================================================\n")

# --- 4. GENERATE & SAVE COMPARISON BAR GRAPH ---
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="Blues_d")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1.05)
for index, row in results_df.iterrows():
    plt.text(index, row['Accuracy'] + 0.02, f"{row['Accuracy']*100:.1f}%", ha='center')

plt.tight_layout()
plt.savefig("accuracy_comparison.png")
print("Saved bar chart as 'accuracy_comparison.png'")

# --- 5. GENERATE & SAVE CONFUSION MATRIX FOR LOGISTIC REGRESSION ---
cm = confusion_matrix(y_test, models["Logistic Regression"].predict(X_test_vec))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Logistic Regression)')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion matrix as 'confusion_matrix.png'")