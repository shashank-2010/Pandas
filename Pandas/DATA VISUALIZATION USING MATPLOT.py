#!/usr/bin/env python
# coding: utf-8

# # 1. Exercise on Average IQ data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


df = pd.read_csv('D:\data analysis\dataset for analysis\Avg_IQ.csv')
df.head(30)


# In[3]:


a = df.groupby('Continent')[['Continent','Nobel Prices']].sum()
df_nobel = a.reset_index()
print(df_nobel)

sn.barplot(x='Continent', y = 'Nobel Prices', data = df_nobel)
plt.title("Nobel Prize per Continent")


# In[10]:


df_literacy = df.groupby("Continent")[['Continent','Literacy Rate']].mean()
df_literacy_conti = df_literacy.reset_index()
df_literacy_conti

sn.barplot(x = 'Continent', y = 'Literacy Rate', data = df_literacy_conti)


# In[12]:


sn.scatterplot(x = 'Continent',y = 'Literacy Rate', data = df_literacy_conti);


# In[25]:


df_conti = df.groupby('Continent')[['Country','Literacy Rate']]
df_conti_asia = df_conti.get_group('Asia').head(10)
df_conti_asia.reset_index()

sn.scatterplot(x='Country',y='Literacy Rate', data = df_conti_asia)


# In[35]:


df_conti = df.groupby('Continent')[['Country','Literacy Rate']]
df_conti_asia = df_conti.get_group('Asia').head(5)
df_conti_asia.reset_index()

df_conti = df.groupby('Continent')[['Country','Literacy Rate']]
df_conti_Europe = df_conti.get_group('Europe').head(5)
df_conti_Europe.reset_index()

df_concat = pd.concat([df_conti_asia, df_conti_Europe],axis= 0 , keys = ['Asia','Europe'])
df_concat


# # 2. Exercise on data using URL

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"

df_data = pd.read_csv(url)
df_data.info()


# In[4]:


df_data


# In[8]:


#Draw a barplot to show the number of persons having chd in comparison to those having family history or not.

filter_criteria = df_data[(df_data['chd']==1)]

df_filter = filter_criteria[['famhist']].value_counts()
a =df_filter.reset_index()
print(a)

ax = sns.barplot(x='famhist',y= 0 ,data = a);
ax.set_xlabel("Family History of Disease")
ax.set_ylabel("Number of chd")
plt.title('chd patients with or without family history')


# In[81]:


#DOES AGE AND SBP HAS ANY CORRELATION 
a_corr = df_data['age'].corr(df_data['sbp'])
print(a_corr)

sns.regplot(x='age',y='sbp',data=df_data);


# In[87]:


#COMPARING DISTRIBUTION OF TOBACCO CONSUMPTION AMONG CHD AND NON-CHD

sns.distplot(df_data[df_data['chd']==1]['tobacco'], color ='g', label = 'Tobacco with chd')

sns.distplot(df_data[df_data['chd']==0]['tobacco'], color ='r', label = 'Tobacco without chd')

plt.legend();


# In[90]:


#How are sbp,obesity,age,ldl correlated?

condition_col = ['sbp','obesity','age','ldl']

sns.pairplot(df_data[condition_col], height=2);


# In[6]:


#Derive a new column called agegroup from age column where persons falling in different age ranges are categorised.

import pandas as pd

# Dictionary creation
age_ranges = {"young": (0, 18),"adult": (18, 40),"mid": (40, 65),"old": (65, None)}

#new column creation named "AgeGroup"
def categorize_age(age):
    for i, (min_age, max_age) in age_ranges.items(): #items is used because the records are in dictionary
        if min_age <= age < max_age:
            return i
    return None

df_data["AgeGroup"] = df_data["age"].apply(categorize_age)

df_data.iloc[50:70]


# In[7]:


#Find out the number of CHD cases in different age categories. Draw a barplot.
filter_criteria = df_data[(df_data['chd']==1)]

df_chd_agegroup = filter_criteria[['AgeGroup']].value_counts()
a_data = df_chd_agegroup.reset_index()
print(a_data)

axis = sns.barplot(x = 'AgeGroup', y = 0, data= a_data)
axis.set_xlabel("AgeGroup")
axis.set_ylabel('Number of chd patient')


# In[ ]:





# In[ ]:




