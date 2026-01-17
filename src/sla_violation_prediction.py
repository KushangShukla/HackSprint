#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# In[2]:


DATA_PATH = r"D:\Projects\Hack\data\complaints_nlp_processe.csv"
df = pd.read_csv(DATA_PATH)

df.head


# In[3]:


df["sla_breached"] = df["timely_response"].apply(
    lambda x: 0 if x == "Yes" else 1
)

df["sla_breached"].value_counts(normalize=True)


# In[4]:


df = pd.read_csv(r"D:\Projects\Hack\data\complaints_nlp_processe.csv")

df["processed_text"] = df["processed_text"].fillna("")
df = df[df["processed_text"].str.strip() != ""]

df.head()


# In[9]:


df = df.sample(50000, random_state=42).reset_index(drop=True)


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


# In[9]:


import nltk
nltk.download("vader_lexicon")


# In[11]:


import pandas as pd

df = pd.read_csv(r"D:\Projects\Hack\data\complaints_nlp_processe.csv")

df["processed_text"] = df["processed_text"].fillna("")
df = df[df["processed_text"].str.strip() != ""]

# Optional: sample if dataset is huge
df = df.sample(50000, random_state=42).reset_index(drop=True)

df.head()


# In[12]:


from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    return score

df["sentiment_score"] = df["processed_text"].apply(vader_sentiment)


# In[13]:


def sentiment_label(score):
    if score <= -0.05:
        return -1
    elif score >= 0.05:
        return 1
    else:
        return 0

df["sentiment_value"] = df["sentiment_score"].apply(sentiment_label)


# In[14]:


df["sentiment_value"].value_counts()


# In[15]:


df[["processed_text", "sentiment_score", "sentiment_value"]].head()


# In[16]:


df.to_csv("D:\Projects\Hack\data\complaints_with_sentiment.csv", index=False)
print("Saved fast sentiment dataset")

