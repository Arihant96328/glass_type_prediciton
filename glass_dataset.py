#!/usr/bin/env python
# coding: utf-8

# In[52]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[53]:


#importing the glass dataset
df=pd.read_csv(r'F:\glass.csv')
df.head()


# In[54]:


print(df.shape)
print(df.info())


# In[55]:


sns.boxplot(df['Type'],df['RI'] )


# In[56]:


df.describe()


# In[57]:


#no. of types of glass in our set
df['Type'].unique()


# In[58]:


#countplot of the number of glass
sns.countplot(data=df,x='Type')


# In[59]:


plt.figure(figsize=(14,8))
plt.subplot(1,2,1)
sns.distplot(df['RI'])
plt.subplot(1,2,2)
sns.boxplot(df['RI'])


# In[60]:


plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(df['Na'])
plt.subplot(1,2,2)
sns.boxplot(df['Na'])


# In[61]:


#corelation between the features.Value close to 1 directly proportionality and vice versa
df.corr().style.background_gradient().set_precision(2)


# In[62]:


train=df.drop(columns='Type')
test=df['Type']


# In[69]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.2,random_state=4)
nb.fit(x_train,y_train)
pred=nb.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))


# In[70]:


class_names=['1','2','3','5','6','7']
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,pred)
df=pd.DataFrame(matrix,index=class_names,columns=class_names)
sns.heatmap(df,annot=True,cmap='Blues',cbar='None')
plt.title('CM')
plt.tight_layout()
plt.show()


# In[66]:


#Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='entropy',max_depth=4)
tree.fit(x_train,y_train)
pred=tree.predict(x_test)
print(accuracy_score(y_test,pred))


# In[67]:


#using xgboost algorithm
import xgboost
model1=xgboost.XGBClassifier()
model1.fit(x_train, y_train)
pred = model1.predict(x_test)
print('Accuracy: ', accuracy_score(y_test, pred))


# In[68]:


class_names=['1','2','3','5','6','7']
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,pred)
df=pd.DataFrame(matrix,index=class_names,columns=class_names)
sns.heatmap(df,annot=True,cmap='Blues',cbar='None')
plt.title('CM')
plt.tight_layout()
plt.show()

