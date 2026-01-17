#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]

    if score <= -0.05:
        value = -1
    elif score >= 0.05:
        value = 1
    else:
        value = 0

    return score, value

