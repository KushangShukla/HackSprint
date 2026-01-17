#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Download & Locate Dataset


# In[1]:


import kagglehub
import os

# Download dataset
path = kagglehub.dataset_download("selener/consumer-complaint-database")

print("Dataset path:", path)
print("Files:", os.listdir(path))


# In[ ]:


# Load Dataset


# In[5]:


import pandas as pd

csv_file = "rows.csv"  # adjust if name differs
df = pd.read_csv(os.path.join(path, csv_file))

df.head()


# In[ ]:


# Rename Columns


# In[6]:


df = df.rename(columns={
    "Date received": "date_received",
    "Product": "product",
    "Issue": "issue",
    "Consumer complaint narrative": "complaint_text",
    "Company": "company",
    "State": "state",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Timely response?": "timely_response",
    "Complaint ID": "complaint_id"
})


# In[ ]:


# Drop Empty Complaint Text


# In[7]:


df = df[
    [
        "complaint_id",
        "complaint_text",
        "product",
        "issue",
        "timely_response"
    ]
]


# In[ ]:


# Keep Only NLP-Relevant Columns


# In[8]:


df = df.dropna(subset=["complaint_text"])
df.shape


# In[ ]:


# Import NLP Libraries


# In[9]:


import re
import string


# In[ ]:


# Clean Raw Text


# In[10]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"\d+", "", text)             # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["complaint_text"].apply(clean_text)

df[["complaint_text", "clean_text"]].head()


# In[ ]:


# Download NLP Resources


# In[13]:


import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")


# In[ ]:


# Tokenize & Remove Stopwords


# In[14]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def tokenize_remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

df["tokens"] = df["clean_text"].apply(tokenize_remove_stopwords)

df[["clean_text", "tokens"]].head()


# In[ ]:


# Lemmatize Tokens


# In[15]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

df["lemmatized_tokens"] = df["tokens"].apply(lemmatize_tokens)

df[["tokens", "lemmatized_tokens"]].head()


# In[ ]:


# Final NLP Text


# In[16]:


df["processed_text"] = df["lemmatized_tokens"].apply(lambda x: " ".join(x))

df[["complaint_text", "processed_text"]].head()


# In[ ]:


# Save for Next Notebooks


# In[21]:


processed_path = "D:\Projects\Hack\data\complaints_nlp_processe.csv"
df.to_csv(processed_path, index=False)

print("Saved processed dataset at:", processed_path)

