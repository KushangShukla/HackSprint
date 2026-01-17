#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Libraries


# In[1]:


import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# In[ ]:


# Loading Data Path


# In[2]:


DATA_PATH = r"D:\Projects\Hack\data\complaints_nlp_processe.csv"
df = pd.read_csv(DATA_PATH)

df.head


# In[ ]:


# SLA Breach Feature Engineering


# In[3]:


df["sla_breached"] = df["timely_response"].apply(
    lambda x: 0 if x == "Yes" else 1
)

df["sla_breached"].value_counts(normalize=True)


# In[ ]:


# Dataset Reload and Text Validation


# In[4]:


df = pd.read_csv(r"D:\Projects\Hack\data\complaints_nlp_processe.csv")

df["processed_text"] = df["processed_text"].fillna("")
df = df[df["processed_text"].str.strip() != ""]

df.head()


# In[ ]:


# Dataset Sampling


# In[9]:


df = df.sample(50000, random_state=42).reset_index(drop=True)


# In[ ]:


# Sentiment Analysis Setup


# In[5]:


from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)

def get_sentiment(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        preds = sentiment_pipeline(
            batch,
            truncation=True,
            max_length=512
        )
        results.extend(preds)
    return results


# In[ ]:


# VADER Lexicon Setup


# In[9]:


import nltk
nltk.download("vader_lexicon")


# In[ ]:


# Dataset Preparation


# In[11]:


import pandas as pd

df = pd.read_csv(r"D:\Projects\Hack\data\complaints_nlp_processe.csv")

df["processed_text"] = df["processed_text"].fillna("")
df = df[df["processed_text"].str.strip() != ""]

# Optional: sample if dataset is huge
df = df.sample(50000, random_state=42).reset_index(drop=True)

df.head()


# In[ ]:


# VADER Sentiment Analysis


# In[12]:


from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    return score

df["sentiment_score"] = df["processed_text"].apply(vader_sentiment)


# In[ ]:


# Sentiment Label Encoding


# In[13]:


def sentiment_label(score):
    if score <= -0.05:
        return -1
    elif score >= 0.05:
        return 1
    else:
        return 0

df["sentiment_value"] = df["sentiment_score"].apply(sentiment_label)


# In[ ]:


# Sentiment Distribution Analysis


# In[14]:


df["sentiment_value"].value_counts()


# In[ ]:


# Sentiment Annotation Preview


# In[15]:


df[["processed_text", "sentiment_score", "sentiment_value"]].head()


# In[ ]:


#Save Enriched Dataset


# In[16]:


df.to_csv("D:\Projects\Hack\data\complaints_with_sentiment.csv", index=False)
print("Saved fast sentiment dataset")


# In[ ]:


# SLA Breach Feature Engineering


# In[17]:


df = pd.read_csv(r"D:\Projects\Hack\data\complaints_with_sentiment.csv")

df["sla_breached"] = df["timely_response"].apply(
    lambda x: 0 if x == "Yes" else 1
)


# In[ ]:


# Hybrid SLA Breach Prediction Model


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib, os

X_text = df["processed_text"]
X_meta = df[["sentiment_value", "sentiment_score"]]
y = df["sla_breached"]

X_text_train, X_text_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    X_text, X_meta, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=10000, min_df=10)
X_text_train_vec = tfidf.fit_transform(X_text_train)
X_text_test_vec = tfidf.transform(X_text_test)

X_train = hstack([X_text_train_vec, X_meta_train.values])
X_test = hstack([X_text_test_vec, X_meta_test.values])

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)


# In[ ]:


# Save SLA Prediction Model


# In[19]:


os.makedirs(r"D:\Projects\Hack\models", exist_ok=True)

joblib.dump(model, r"D:\Projects\Hack\models\sla_violation_model.pkl")
joblib.dump(tfidf, r"D:\Projects\Hack\models\sla_tfidf_vectorizer.pkl")

print("SLA model saved")


# In[ ]:


# SLA Risk Scoring


# In[20]:


X_all_vec = tfidf.transform(df["processed_text"])
X_all = hstack([X_all_vec, df[["sentiment_value", "sentiment_score"]].values])

df["sla_risk"] = model.predict_proba(X_all)[:, 1]


# In[ ]:


# Save SLA Risk Dataset


# In[21]:


df.to_csv(r"D:\Projects\Hack\data\complaints_with_sla_risk.csv", index=False)
print("Saved complaints_with_sla_risk.csv")


# In[ ]:




