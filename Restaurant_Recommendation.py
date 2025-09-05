#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ### Load dataset

# In[14]:


data = pd.read_csv("Dataset.csv")


# In[15]:


data.shape


# In[16]:


data.head()


# In[17]:


data.info()


# ### check misising value

# In[18]:


data.isnull().sum()


# ### Handle missing values

# In[19]:


data.fillna("Unknown", inplace=True)


# ### Combine useful features for recommendations

# In[20]:


# Example: cuisine type + price range + location
if "Cuisines" in data.columns and "Price range" in data.columns:
    data["Features"] = data["Cuisines"].astype(str) + " " + data["Price range"].astype(str)
else:
    raise ValueError("Dataset must contain 'Cuisines' and 'Price range' columns")


# ### Vectorize text features

# In[21]:


vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(data["Features"])


# ### Compute similarity matrix

# In[22]:


similarity_matrix = cosine_similarity(feature_matrix)


# ### Recommendation function

# In[23]:


def recommend_restaurants(cuisine_pref, price_pref, top_n=5):
    # Create preference string
    user_pref = cuisine_pref + " " + str(price_pref)
    user_vec = vectorizer.transform([user_pref])

    # Compute similarity with all restaurants
    scores = cosine_similarity(user_vec, feature_matrix).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]

    return data.iloc[top_indices][["Restaurant Name", "Cuisines", "Price range", "Aggregate rating"]]


# ### Test recommendation system

# In[24]:


user_cuisine = "Italian"
user_price = 2   # Example: price range 1–4
recommendations = recommend_restaurants(user_cuisine, user_price)
print("Recommended Restaurants:\n", recommendations)


# In[ ]:




