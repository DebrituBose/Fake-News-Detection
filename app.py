import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Fake News Detector", page_icon="🔍", layout="centered")

# --- CACHE DATA & MODEL TRAINING FOR SPEED ---
@st.cache_resource
def load_and_train_model():
    # Define all possible paths where True.csv and Fake.csv might be sitting
    possible_paths = [
        {"true": "True.csv", "fake": "Fake.csv"},                                # Root folder
        {"true": "../True.csv", "fake": "../Fake.csv"},                          # One folder up
        {"true": "data/True.csv", "fake": "data/Fake.csv"},                      # Inside data folder
        {"true": "/workspaces/Fake-News-Detection/True.csv",                     # Absolute workspace path
         "fake": "/workspaces/Fake-News-Detection/Fake.csv"}
    ]
    
    true_df, fake_df = None, None
    
    # Loop through paths until we find where the files are hiding
    for paths in possible_paths:
        if os.path.exists(paths["true"]) and os.path.exists(paths["fake"]):
            true_df = pd.read_csv(paths["true"])
            fake_df = pd.read_csv(paths["fake"])
            break
            
    # Ultimate fail-safe if files are completely missing from the workspace
    if true_df is None or fake_df is None:
        st.error("❌ Crucial dataset files ('True.csv' and 'Fake.csv') not found in your Codespace workspace directory structure. Please check your file explorer sidebar panel.")
        st.stop()
        
    true_df['label'] = 0
    fake_df['label'] = 1
        
    # Using 20% sample chunk so the web app loads in seconds instead of freezing
    df = pd.concat([true_df, fake_df], axis=0).sample(frac=0.2, random_state=42).reset_index(drop=True)
    
    X, y = df['text'], df['label']
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_vectorized, y)
    
    return vectorizer, model

# Initialize components
vectorizer, model = load_and_train_model()

# --- USER INTERFACE (UI) ---
st.title("🔍 AI Fake News Classification Engine")
st.markdown("This application uses a trained **TF-IDF + Linear Matrix Classifier** to analyze news text parameters and evaluate the statistical probability of structural misinformation.")
st.write("---")

# Text Area Input
user_input = st.text_area("📋 Paste the full text of the news article below:", height=250, 
                          placeholder="Type or paste text structure here...")

# Evaluation Button
if st.button("🚀 Analyze Article Authenticity", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please provide a valid text string to run verification.")
    else:
        with st.spinner("Analyzing linguistic features..."):
            vectorized_input = vectorizer.transform([user_input])
            prediction = model.predict(vectorized_input)[0]
            probabilities = model.predict_proba(vectorized_input)[0]
            
            st.write("### 📊 Classification Report Results")
            
            if prediction == 1:
                st.error(f"🚨 **ALERT: Classified as FAKE NEWS**")
                st.metric(label="Classification Confidence", value=f"{probabilities[1]*100:.2f}% sure it is fabricated.")
                st.progress(float(probabilities[1]))
            else:
                st.success(f"✅ **VERIFIED: Classified as REAL NEWS**")
                st.metric(label="Authentication Confidence", value=f"{probabilities[0]*100:.2f}% sure it is authentic.")
                st.progress(float(probabilities[0]))