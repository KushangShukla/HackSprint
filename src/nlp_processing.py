#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Libraries


# In[3]:


import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# In[ ]:


#Load and Preview Dataset


# In[4]:


DATA_PATH = "D:\Projects\Hack\data\complaints_nlp_processe.csv"

df = pd.read_csv(DATA_PATH)

df.head()


# In[ ]:


# Feature–Target Separation


# In[5]:


X = df["processed_text"]
y = df["product"]

print("Number of samples:", len(df))
print("Number of categories:", y.nunique())


# In[ ]:


# Remove Rare Classes


# In[6]:


min_samples = 500
valid_products = y.value_counts()[y.value_counts() >= min_samples].index

df = df[df["product"].isin(valid_products)]

X = df["processed_text"]
y = df["product"]

print("Filtered samples:", len(df))
print("Remaining categories:", y.nunique())


# In[ ]:


# Train–Test Split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[ ]:


# Text Data Cleaning


# In[8]:


# Replace NaN with empty string
df["processed_text"] = df["processed_text"].fillna("")

# Remove rows where processed_text is empty
df = df[df["processed_text"].str.strip() != ""]

print("Remaining samples:", len(df))


# In[ ]:


# Refresh Feature–Target Mapping


# In[9]:


X = df["processed_text"]
y = df["product"]


# In[ ]:


# Final Train–Test Split


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[ ]:


# TF-IDF Vectorization


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# In[ ]:


# Model Training


# In[12]:


model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)


# In[ ]:


# Model Evaluation


# In[13]:


y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# In[ ]:


# Model Saving


# In[14]:


os.makedirs(r"D:\Projects\Hack\models", exist_ok=True)

joblib.dump(model, r"D:\Projects\Hack\models\complaint_category_model.pkl")
joblib.dump(tfidf, r"D:\Projects\Hack\models\tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully")


# In[ ]:


# Data Reload and Null Handling


# In[15]:


import pandas as pd

DATA_PATH = "D:\Projects\Hack\data\complaints_nlp_processe.csv"
df = pd.read_csv(DATA_PATH)

df = df[df["processed_text"].notna()]
df.head()


# In[ ]:


# Sentiment Analysis Model Setup


# In[19]:


from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)


# In[ ]:


# Batched Sentiment Inference Function


# In[20]:


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


# Sentiment Prediction on Sample Data


# In[21]:


sample_df = df.sample(5000, random_state=42)

sentiments = get_sentiment(sample_df["processed_text"].tolist())

sample_df["sentiment_label"] = [s["label"] for s in sentiments]
sample_df["sentiment_score"] = [s["score"] for s in sentiments]

sample_df.head()


# In[ ]:


# Deep Learning Dependencies (LSTM Setup)


# In[22]:


import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# In[ ]:


# Data Loading and Text Cleaning


# In[24]:


df = pd.read_csv(r"D:\Projects\Hack\data\complaints_nlp_processe.csv")

df["processed_text"] = df["processed_text"].fillna("")
df = df[df["processed_text"].str.strip() != ""]


# In[ ]:


# Dataset Sampling


# In[25]:


df = df.sample(80000, random_state=42).reset_index(drop=True)


# In[ ]:


# Encode product labels


# In[26]:


labels = df["product"].astype("category")
label_to_index = dict(enumerate(labels.cat.categories))
index_to_label = {v: k for k, v in label_to_index.items()}

y = labels.cat.codes
y = to_categorical(y)


# In[ ]:


# Text Tokenization and Padding


# In[27]:


MAX_WORDS = 20000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["processed_text"])

sequences = tokenizer.texts_to_sequences(df["processed_text"])
X = pad_sequences(sequences, maxlen=MAX_LEN)


# In[ ]:


#Train–Test Split for LSTM Model


# In[31]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# In[ ]:


# LSTM Model Architecture & Compilation


# In[29]:


model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(y.shape[1], activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# In[ ]:


# LSTM Model Training


# In[32]:


history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128
)


# In[ ]:


# LSTM Model Evaluation


# In[33]:


loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)


# In[ ]:


# Save Deep Learning Model and Assets


# In[36]:


import os
import joblib

os.makedirs(r"D:\Projects\Hack\models\tf", exist_ok=True)

# Save TensorFlow model
model.save(r"D:\Projects\Hack\models\tf\complaint_category_tf_model.keras")

# Save tokenizer & label mapping
joblib.dump(tokenizer, r"D:\Projects\Hack\models\tf\tf_tokenizer.pkl")
joblib.dump(index_to_label, r"D:\Projects\Hack\models\tf\label_mapping.pkl")

print("TensorFlow model, tokenizer, and label mapping saved successfully")


# In[ ]:




