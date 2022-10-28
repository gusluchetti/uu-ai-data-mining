#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="dark")


# In[2]:


pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 20)
og_eclipse_df = pd.read_csv('./data/eclipse-metrics-packages-2.0.csv', delimiter=";")
og_eclipse_df


# In[3]:


# getting target label
y = og_eclipse_df["post"]
y.head()


# In[4]:


y.describe()


# In[5]:


# getting predictor variables
x = og_eclipse_df.drop(columns=["post"]).iloc[:, 2:43]
x.head()


# In[6]:


x.describe()


# ## Data Analysis

# In[8]:


sns.jointplot(data=og_eclipse_df, x="ACD_avg", y="VG_avg", hue="post")


# In[ ]:


sns.pairplot(data=og_eclipse_df, hue="post")

