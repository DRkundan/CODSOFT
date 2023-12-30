#!/usr/bin/env python
# coding: utf-8

# ###  CODSOFT TASK 4
# 
# ###  SALES PREDICTION USING PYTHON
# 
# #### NAME : Dr Kundan k Mane
#  #### BATCH : A19
# #### DOMAIN : DataScience
# 
# ##### Aim : Build a  model to predict the sales of Advertising in Newspaper, Radio and Television

# In[2]:


from IPython.display import Image
Image(url="https://houseofhrvst.com/wp-content/uploads/2020/09/5c57efb0c14b9-750x460.jpg", height=400,width=1000)


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[5]:


data=pd.read_csv(r'C:\Kundan files\CODSOFT KUNDAN\task4\advertising.csv')
data


# In[7]:


data.info()


# In[6]:


data.describe()


# In[ ]:





# ##### from the mean value TV has the highest expensable platform for sales. Whereas Radio has the lowest expensable platform for sales.

# In[7]:


from matplotlib import style


# In[8]:


style.use('ggplot')
fig=plt.figure(figsize=(15,5))
fig.suptitle('Scatter Plot of TV, Radio, and Newspaper vs. Sales')
plt.subplot(1,3,1)
plt.scatter(data['TV'], data['Sales'],color='royalblue')
plt.xlabel('Advertising in TV')
plt.ylabel('Sales')
plt.subplot(1,3,2)
plt.scatter(data['Radio'], data['Sales'],color='orange')
plt.xlabel('Advertising in Radio')
plt.ylabel('Sales')
plt.subplot(1,3,3)
plt.scatter(data['Newspaper'], data['Sales'],color='green')
plt.xlabel('Advertising in Newspaper')
plt.ylabel('Sales')
plt.show()
plt.show()


# In[9]:


plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
sns.distplot(data['TV'])
plt.xlabel('TV')
plt.subplot(1,3,2)
sns.distplot(data['Newspaper'])
plt.xlabel('Radio')
plt.subplot(1,3,3)
sns.distplot(data['Newspaper'])
plt.xlabel('Newspaper')
plt.show()


# ### TV as high variance in their density comparitively to Radio and Newspaper.

# In[10]:


correlation_matrix=data.corr()
plt.figure(figsize=(8,5))  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# ##### From this correlation map we understand that TV has highly correlated. so, I choose x data as TV and y as Sales.

# In[11]:


x=data[['TV']]
y=data[['Sales']]


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[14]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[15]:


def class_evaluation(y_test,y_pred):
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    
    print('Mean absolute error :', mae)
    print('Mean squared error :', mse)
    print('R2_ score :', r2)
    print('Root Mean squared error',rmse)
    
def train_test_score(model):
    print('Training Score',model.score(x_train,y_train))
    print('Testing score',model.score(x_test,y_test))


# ### Build a model Using Ada boost regressor

# In[16]:


from sklearn.ensemble import AdaBoostRegressor
adr=AdaBoostRegressor()
adr.fit(x_train,y_train)
ypred_adr=adr.predict(x_test)
train_test_score(adr)


# In[17]:


class_evaluation(y_test,ypred_adr)


# ### Prediction of sales using Adaboost Regressor

# In[18]:


data['Sales_predict']=adr.predict(x)
data.head()


# ####  Regression plot for Actual sales rate and predicted sales rate.

# In[19]:


plt.figure(figsize=(15,5))
sns.regplot(x=data['Sales'],y=data['Sales_predict'],color='blue',line_kws={"color": "green"})
plt.xlabel('Actual Sales Rate',color='blue')
plt.ylabel('Predicted Sales Rate',color='red')
plt.show()


# #### comparing the denisty of Actual sales rate and predicted sales rate using Distplot.

# In[20]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(data['Sales'],color='blue',hist_kws={'density': True})
plt.xlabel('Actual sales rate')
plt.subplot(1,2,2)
sns.distplot(data['Sales_predict'],color='green',hist_kws={'density': True})
plt.xlabel('Predicted sales rate')
plt.show()


# In[ ]:




