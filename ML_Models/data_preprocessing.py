#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


# ### Import Dataset

# In[2]:


data_set = pd.read_csv(r"I:\1.CSE\Thesis\LCPUMLA\Datasets\lung_cancer1.csv")


# ### Show dataset's row and column

# In[3]:


data_set.shape


# ### Dataset's head 

# In[4]:


data_set.head()


# ### Data types of dataset columns 

# In[5]:


data_set.dtypes


# ### Check duplicate row 

# In[6]:


data_set.loc[data_set.duplicated(), :]


# ### Check null value

# In[7]:


data_set.isnull().values.any()


# ### Result percentage of dataset

# In[8]:


n_low = len(data_set.loc[data_set['Level'] == "Low"])
n_medium = len(data_set.loc[data_set['Level'] == "Medium"])
n_high = len(data_set.loc[data_set['Level'] == "High"])

print ("Number of Low Cases: {0} ({1:2.2f}%)".format(n_low, (n_low / (n_low + n_medium + n_high)) * 100))
print ("Number of Medium Cases: {0} ({1:2.2f}%)".format(n_medium, (n_medium / (n_low + n_medium + n_high)) * 100))
print ("Number of High Cases: {0} ({1:2.2f}%)".format(n_high, (n_high / (n_low + n_medium + n_high)) * 100))


# ### Dataset correlation heatmap

# In[9]:


def correlation_heatmap(data_set, size):
    correlation = data_set.corr()
    
    # print correlation
    # Dividing the plot into subplots for increasing size of plots
    fig, heatmap = plt.subplots(figsize=(size, size))
    
    # show heatmap 
    heatmap.matshow(correlation)
    
    # Adding xticks and yticks
    # plt.xticks(range(len(correlation.columns)), correlation.columns, color="red", fontsize=7 )
    plt.yticks(range(len(correlation.columns)), correlation.columns, color="blueviolet", fontsize=30)
    
    # Displaying the graph
    plt.show()


# In[10]:


correlation_heatmap(data_set, 25)


# ### Dataset preprocessing

# In[11]:


#change level column's value into number
change_level = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

target_class = data_set["Level"].values

data_set["Level"] = data_set["Level"].map(change_level)

# delete patient id 
del data_set["Patient Id"]


# ### Dataset after preprocessing

# In[12]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data_set


# ### Features values for data splitting and features selection 

# In[13]:


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

F_names= data_set.iloc[:,:-1]
P_names= data_set.iloc[:,:24]


# ### Features Selection 

# In[14]:


# create model for select best feature 
select_best_feature_model = SelectKBest(score_func=f_classif)
select_best_feature_model.fit(F,P.ravel())


# In[15]:


columns_score = pd.DataFrame(select_best_feature_model.scores_, columns=["Score Value"])
columns_name = pd.DataFrame(F_names.columns, columns=["Feature Name"])
column_score_with_label = pd.concat([columns_name,columns_score],axis=1)


# In[16]:


# columns score
column_score_with_label


# In[17]:


# top features
column_score_with_label.nlargest(18, "Score Value")


# ### Dataset spliting for all features 

# In[18]:


# Saving 30% for testing
split_test_size = 0.30

# Splitting
F_train, F_test, P_train, P_test = train_test_split(F, P, test_size = split_test_size, random_state = 1)

#check spliting is accurate or not 
print("{0:0.2f}% in training set".format((len(F_train)/len(data_set.index)) * 100))
print("{0:0.2f}% in test set".format((len(F_test)/len(data_set.index)) * 100)) 


# ### Dataset spliting for selected features 

# In[19]:


selected_features_col = ['AirPollution','Alcoholuse','DustAllergy','OccuPationalHazards','GeneticRisk','BalancedDiet',
                         'Obesity','PassiveSmoker','ChestPain','CoughingofBlood', 'Smoking', 'Fatigue' ,
                         'chronicLungDisease', 'ShortnessofBreath','FrequentCold','Wheezing','ClubbingofFingerNails','WeightLoss']

F_selected = data_set[selected_features_col].values

# Splitting
F_selec_train, F_selec_test, P_selec_train, P_selec_test = train_test_split(F_selected, P, test_size = split_test_size, random_state = 1)


# In[20]:


#check spliting is accurate or not 
print("{0:0.2f}% in training set".format((len(F_selec_train)/len(data_set.index)) * 100))
print("{0:0.2f}% in test set".format((len(F_selec_test)/len(data_set.index)) * 100)) 

