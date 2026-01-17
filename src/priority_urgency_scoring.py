#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Libraries


# In[2]:


import pandas as pd
import numpy as np

df = pd.read_csv("D:\Projects\Hack\data\complaints_with_sentiment.csv")

df["processed_text"] = df["processed_text"].fillna("")
df = df[df["processed_text"].str.strip() != ""]

df.head()


# In[ ]:


# Text Length Feature Engineering


# In[3]:


# Complaint length
df["text_length"] = df["processed_text"].apply(lambda x: len(x.split()))

# Normalize length
df["text_length_norm"] = (df["text_length"] - df["text_length"].min()) / (
    df["text_length"].max() - df["text_length"].min()
)


# In[ ]:


# Load SLA Risk Dataset


# In[6]:


df = pd.read_csv(r"D:\Projects\Hack\data\complaints_with_sla_risk.csv")


# In[ ]:


# SLA Risk Inference Pipeline


# In[12]:


import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

sla_model = joblib.load(r"D:\Projects\Hack\models\sla_violation_model.pkl")
sla_vectorizer = joblib.load(r"D:\Projects\Hack\models\sla_tfidf_vectorizer.pkl")

# Text features
X_text_vec = sla_vectorizer.transform(df["processed_text"])

# Meta features (must match training order)
X_meta = df[["sentiment_value", "sentiment_score"]].values

# Combine features EXACTLY like training
X_sla = hstack([X_text_vec, X_meta])

# Predict SLA risk
df["sla_risk"] = sla_model.predict_proba(X_sla)[:, 1]


# In[ ]:


# Save SLA Risk Output


# In[14]:


df.to_csv(r"D:\Projects\Hack\data\complaints_with_sla_risk.csv", index=False)
print("Saved complaints_with_sla_risk.csv")


# In[ ]:


# Text Length Calculation


# In[16]:


df["text_length"] = df["processed_text"].apply(lambda x: len(x.split()))


# In[ ]:


# Text Length Normalization


# In[17]:


df["text_length_norm"] = (
    df["text_length"] - df["text_length"].min()
) / (
    df["text_length"].max() - df["text_length"].min()
)


# In[ ]:


# Handle Missing Values (Text Length)


# In[18]:


df["text_length_norm"] = df["text_length_norm"].fillna(0)


# In[ ]:


#Complaint Priority Assignment


# In[19]:


df["priority_score"] = (
    0.4 * (df["sentiment_value"] == -1).astype(int) +
    0.3 * df["sla_risk"] +
    0.3 * df["text_length_norm"]
)


# In[ ]:


# Priority Level Assignment


# In[20]:


def assign_priority(score):
    if score >= 0.75:
        return "Critical"
    elif score >= 0.5:
        return "High"
    elif score >= 0.25:
        return "Medium"
    else:
        return "Low"

df["priority_level"] = df["priority_score"].apply(assign_priority)


# In[ ]:


# Saved Model


# In[21]:


df.to_csv("D:\Projects\Hack\data\complaints_with_priority.csv", index=False)
print("Priority model completed successfully")


# In[ ]:




