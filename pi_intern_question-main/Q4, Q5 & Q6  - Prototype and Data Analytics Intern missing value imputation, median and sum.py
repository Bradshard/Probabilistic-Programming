#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import statistics


# In[2]:


data = pd.read_csv("country_vaccination_stats.csv")
data


# In[3]:


unique_countries = data["country"].unique()
unique_countries

data.loc[data["country"] == "Argentina"]


# In[4]:


#df.daily_vaccinations.fillna(df.Farheit, inplace=True)

#pd.Index(data.loc[data["country"] == "Argentina"])


# In[5]:


# to try it first time

df = data["daily_vaccinations"].groupby(data["country"]).unique()
df["Argentina"] = np.nan_to_num(df["Argentina"], nan = np.nanmin(df["Argentina"]))
df["Argentina"]


# In[6]:


df = data["daily_vaccinations"].groupby(data["country"]).unique()

for i in range(0,data["country"].nunique()):
    if len(df[i])>1:
        df[i] = np.nan_to_num(df[i], nan = np.nanmin(df[i]))
    else:
        df[i] = np.nan_to_num(df[i], nan = 0)

df


# In[7]:


w = 0 
a = []
a = data[data["daily_vaccinations"].isnull()].index.tolist()

a[w]


# In[8]:



for i in data["country"].unique():
    if (len(data.loc[data["country"] == i,"daily_vaccinations"])>1): 
        data["daily_vaccinations"][a[w]] = np.nanmin(data.loc[data["country"] == i,"daily_vaccinations"])
    else:
        data["daily_vaccinations"][a[w]] = np.nan_to_num(data["daily_vaccinations"][a[w]], nan = 0)
    w = w + 1
w = 0    


# In[9]:


data


# In[10]:


#data.loc[data["country"] == "Argentina","daily_vaccinations"][0] = np.nan_to_num(data.loc[data["country"] == "Argentina","daily_vaccinations"][0],np.nanmin(data.loc[data["country"] == "Argentina","daily_vaccinations"]))


# In[11]:


#data["daily_vaccinations"][0] =  np.nanmin(data.loc[data["country"] == "Argentina","daily_vaccinations"])
#data.loc[data["country"] == "Turkey","daily_vaccinations"][1355]


# In[12]:


#data.fillna(df.groupby("daily_vaccinations").transform("min"), inplace = True)


# In[13]:


## Q5

listing = []
for i in data["country"].unique():
    listing.append(statistics.median(data.loc[data["country"] == i,"daily_vaccinations"]))

listing


# In[14]:


top_3_median_list = []
new_list = listing.copy()

for i in range(0,3):
    top_3_median_list.append([max(listing),data["country"].unique()[new_list.index(max(listing))]])
    listing.pop(np.argmax(listing))


# In[15]:


top_3_median_list


# In[16]:


sum(data.loc[data["date"] == "1/6/2021","daily_vaccinations"])


# In[ ]:




