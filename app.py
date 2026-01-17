#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================================
# app.py — AI Public Complaint Intelligence
# ==========================================

import streamlit as st
import joblib
import tensorflow as tf
from scipy.sparse import hstack
import re

# ---------- NLP / SENTIMENT ----------
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# ---------- NLTK SETUP ----------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# ---------- TEXT PREPROCESSING ----------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)

# ---------- SENTIMENT ----------
def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score <= -0.05:
        value = -1
    elif score >= 0.05:
        value = 1
    else:
        value = 0
    return score, value

# ---------- PRIORITY SCORING ----------
def compute_priority(sentiment_value, sla_risk, processed_text):
    length_norm = min(len(processed_text.split()) / 200, 1)
    score = (
        0.4 * (sentiment_value == -1) +
        0.3 * sla_risk +
        0.3 * length_norm
    )
    if score >= 0.75:
        return "Critical"
    elif score >= 0.5:
        return "High"
    elif score >= 0.25:
        return "Medium"
    else:
        return "Low"

# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(
    page_title="AI Public Complaint Intelligence",
    layout="centered"
)

st.title("🧠 AI-Based Public Complaint Intelligence Platform")
st.write(
    "Analyze public complaints using NLP, ML, and Deep Learning to "
    "classify category, detect sentiment, predict SLA breach risk, "
    "and assign priority."
)

# ==========================================
# LOAD MODELS (CACHED)
# ==========================================

@st.cache_resource
@st.cache_resource
def load_models():
    category_model = joblib.load(r"D:\Projects\Hack\backend\models\complaint_category_model.pkl")
    category_vectorizer = joblib.load(r"D:\Projects\Hack\backend\models\tfidf_vectorizer.pkl")

    sla_model = joblib.load(r"D:\Projects\Hack\backend\models\sla_violation_model.pkl")
    sla_vectorizer = joblib.load(r"D:\Projects\Hack\backend\models\sla_tfidf_vectorizer.pkl")

    tf_model = tf.keras.models.load_model(
        r"D:\Projects\Hack\backend\models\tf\complaint_category_tf_model.keras"
    )

    return category_model, category_vectorizer, sla_model, sla_vectorizer, tf_model


category_model, category_vectorizer, sla_model, sla_vectorizer, tf_model = load_models()


# ==========================================
# USER INPUT
# ==========================================

complaint_text = st.text_area(
    "✍️ Enter complaint text",
    height=180,
    placeholder="The bank has ignored my complaint for months and no one responds..."
)

# ==========================================
# ANALYSIS PIPELINE
# ==========================================

if st.button("🔍 Analyze Complaint"):
    if not complaint_text.strip():
        st.warning("Please enter a complaint.")
    else:
        with st.spinner("Analyzing complaint..."):
            # 1️⃣ Preprocess
            processed = preprocess_text(complaint_text)

            # 2️⃣ Category prediction (ML)
            X_cat = category_vectorizer.transform([processed])
            category = category_model.predict(X_cat)[0]


            # 3️⃣ Sentiment
            sentiment_score, sentiment_value = get_sentiment(processed)

            # 4️⃣ SLA Risk
            X_text = sla_vectorizer.transform([processed])
            X_meta = [[sentiment_value, sentiment_score]]
            X_sla = hstack([X_text, X_meta])

            sla_risk = float(sla_model.predict_proba(X_sla)[0][1])

            # 5️⃣ Priority
            priority = compute_priority(
                sentiment_value,
                sla_risk,
                processed
            )

        # ==================================
        # OUTPUT
        # ==================================

        st.subheader("📊 Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("📂 Complaint Category", category)
            st.metric("😠 Sentiment Score", round(sentiment_score, 3))

        with col2:
            st.metric("⏱ SLA Breach Risk", f"{round(sla_risk * 100, 1)} %")
            st.metric("🚨 Priority Level", priority)

        st.progress(min(sla_risk, 1.0))

        st.success("Analysis complete")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




