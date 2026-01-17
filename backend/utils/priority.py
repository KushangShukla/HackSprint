#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

