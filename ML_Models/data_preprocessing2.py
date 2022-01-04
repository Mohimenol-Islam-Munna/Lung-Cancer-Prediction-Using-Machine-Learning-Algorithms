#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import different library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split


# In[2]:


#Data preprocesing
#import dataset
data_set = pd.read_csv(r"I:\1.CSE\Thesis\LCPUMLA\Datasets\lung_cancer1.csv")


# In[3]:


#show data set's row and column
data_set.shape


# In[4]:


#data set's head 
data_set.head()


# In[5]:


#check null value
data_set.isnull().values.any()


# In[6]:


#data_set correlation heatmap
def correlation_heatmap(data_set, size):
    correlation = data_set.corr()
    
    # print correlation
    # Dividing the plot into subplots for increasing size of plots
    fig, heatmap = plt.subplots(figsize=(size, size))
    
    # show heatmap 
    heatmap.matshow(correlation)
    
    # Adding xticks and yticks
    plt.xticks(range(len(correlation.columns)), correlation.columns, color="red", fontsize=20 )
    plt.yticks(range(len(correlation.columns)), correlation.columns, color="green", fontsize=30)
    
    # Displaying the graph
    plt.show()


# In[7]:


correlation_heatmap(data_set, 25)


# In[8]:


#change level column's value into number
change_level = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

data_set["Level"] = data_set["Level"].map(change_level)


# replace patient id's value 
# p_id = data_set["Patient Id"]

# p_id_len = len(p_id)

# for v in range(p_id_len):
#     p_id[v]=v

# delete patient id 
del data_set["Patient Id"]

data_set.tail(10)


# In[9]:


#data spliting

# features columns name 
features_col = ['Age','Gender','AirPollution','Alcoholuse','DustAllergy',
                'OccuPationalHazards','GeneticRisk','chronicLungDisease','BalancedDiet','Obesity',
                'Smoking','PassiveSmoker','ChestPain','CoughingofBlood','Fatigue','WeightLoss',
                'ShortnessofBreath','Wheezing','SwallowingDifficulty','ClubbingofFingerNails',
                'FrequentCold','DryCough','Snoring']

# prediction or result columns name 
predict_col = ['Level']

F = data_set[features_col].values
P = data_set[predict_col].values

# Saving 30% for testing
split_test_size = 0.30

# Splitting
F_train, F_test, P_train, P_test = train_test_split(F, P, test_size = split_test_size, random_state = 1)

#check spliting is accurate or not 
print("{0:0.2f}% in training set".format((len(F_train)/len(data_set.index)) * 100))
print("{0:0.2f}% in test set".format((len(F_test)/len(data_set.index)) * 100)) 

