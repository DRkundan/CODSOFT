#!/usr/bin/env python
# coding: utf-8

# ###  CODSOFT TASK 2
# 
# ###  MOVIE RATING PREDICTION WITH PYTHON
# 
# #### NAME : Dr kundan kumar mane
# #### BATCH : A19
# #### DOMAIN : DataScience
# 
# ##### Aim :  model predicting the rating of the movie with features like Genre, Actors, Directors, Movie names , Year

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")


# In[4]:


import pandas as pd
encodings = ['utf-8', 'latin1', 'ISO-8859-1']

for encoding in encodings:
    try:
        df = pd.read_csv(r'D:\CODSOFT KUNDAN\TASK2\archive\IMDb Movies India.csv', encoding=encoding)
        print(df)
        break  
    except UnicodeDecodeError:
        print(f"Failed to read with encoding '{encoding}'")


# In[3]:


df.shape


# In[4]:


df.duplicated().value_counts()


# In[5]:


df.drop_duplicates(inplace=True)


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.dropna(inplace=True)


# In[9]:


df.shape


# In[10]:


df.isnull().value_counts()


# In[11]:


df['Year'] = df['Year'].str.extract('(\d+)').astype(int)


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


lb=LabelEncoder()
df['Votes']=lb.fit_transform(df['Votes'])
df['Duration']=lb.fit_transform(df['Duration'])


# In[14]:


df.describe()


# In[15]:


df.groupby('Year')['Votes'].mean().sort_values(ascending=True)


# In[16]:


plt.figure(figsize=(17,5))
sns.barplot(x='Year',y='Votes',data=df)
plt.xlabel('Year of the movies released',fontsize='x-large',color='magenta')
plt.ylabel('voting to movies based on year',fontsize='x-large',color='purple')
plt.xticks(rotation=90)
plt.show()


# In[17]:


df['Rating'].unique()


# In[18]:


df['Rating']=df['Rating'].round().astype(int)


# In[19]:


df['Rating'].unique()


# In[20]:


year_count = df.groupby('Year').size().reset_index(name='count')

plt.figure(figsize=(20, 8))
bars = plt.bar(year_count['Year'], year_count['count'], color='blue')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.title('Number of movie releases per year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(range(1930, 2022), rotation=90)
plt.show()


# In[21]:


top_10_genres= df.groupby('Genre').agg({'Rating':'mean'}).sort_values(by='Rating', ascending=False).reset_index().head(10)

plt.figure(figsize=(15,6))
sns.barplot(data=top_10_genres, y='Rating', x='Genre', palette='cool')
plt.title('Top 10 Genre of best movies')
plt.xticks(rotation=90)

for index, value in enumerate(top_10_genres['Rating']):
    plt.text(index, value, str(value), ha='center', va='bottom')
    
plt.show()


# In[22]:


top_10_directors= df.groupby('Director').agg({'Rating':'mean'}).sort_values(by='Rating', ascending=False).reset_index().head(10)

plt.figure(figsize=(15,6))
sns.barplot(data=top_10_directors, y='Rating', x='Director', palette='summer')
plt.title('Top 10 Directors by Highest Rating')

for index, value in enumerate(top_10_directors['Rating']):
    plt.text(index, value, str(value), ha='center', va='bottom')
    
plt.show()


# In[23]:


top_10_movies= df.groupby('Name').agg({'Rating':'mean'}).sort_values(by='Rating', ascending=False).reset_index().head(10)

plt.figure(figsize=(15,6))
sns.barplot(data=top_10_movies, x='Rating', y='Name', palette='Wistia',color='blue')
plt.title('Top 10 Movies')

for index, value in enumerate(top_10_movies['Rating']):
    plt.text(value, index, f'{value:.2f}', ha='left', va='center')
    
plt.show()


# In[24]:


fig=plt.figure(figsize=(16,6))
fig.suptitle('Relationship between rating and votes')
plt.subplot(1,2,1)
sns.distplot(df[['Rating']]).set_title('Rating of movies')
plt.subplot(1,2,2)
sns.distplot(df[['Votes']]).set_title('vote counts for ratings')
plt.show()


# In[25]:


df.drop(['Actor 1','Actor 2','Actor 3','Director','Name'],axis=1,inplace=True)


# In[26]:


df['Genre']=lb.fit_transform(df['Genre'])


# In[27]:


correlation_matrix=df.corr()
plt.figure(figsize=(8,5))  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[28]:


x=df.drop(['Rating'],axis=1)
y=df['Rating']
print(type(x))
print(type(y))
print(x.shape)
print(y.shape)


# In[29]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[30]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def eval_class(y_test,y_pred):
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    r2=r2_score(y_test,y_pred)
    
    print('Mean Absolute Error :', mae)
    print('Mean Squared Error : ',mse)
    print('Root of Mean Squared Eror :', rmse)
    print('R2_Score :', r2)
    
def model(train_test):
    print('Training score: ',train_test.score(x_train,y_train))
    print('Testing Score',train_test.score(x_test,y_test))
    


# In[31]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression(n_jobs=1000)
lin_reg.fit(x_train,y_train)
ypred_lin_reg=lin_reg.predict(x_test)
model(lin_reg)


# In[32]:


eval_class(y_test,ypred_lin_reg)


# In[33]:


from sklearn.tree import DecisionTreeRegressor
dtree_reg=DecisionTreeRegressor(max_depth=200,max_features=100,min_samples_split=500,min_samples_leaf=500,random_state=42,criterion='squared_error')
dtree_reg.fit(x_train,y_train)
ypred_dtree_reg=dtree_reg.predict(x_test)
model(dtree_reg)


# In[34]:


eval_class(y_test,ypred_dtree_reg)


# In[35]:


from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor(max_depth=10,max_features=100,n_estimators=1500,learning_rate=0.05,subsample=0.8,min_samples_split=500,min_samples_leaf=500,random_state=42)
gbr.fit(x_train,y_train)
ypred_gbr=gbr.predict(x_test)
model(gbr)


# In[36]:


eval_class(y_test,ypred_gbr)


# In[37]:


from sklearn.ensemble import AdaBoostRegressor
adr=AdaBoostRegressor()
adr.fit(x_train,y_train)
ypred_adr=adr.predict(x_test)
model(adr)


# In[38]:


eval_class(y_test,ypred_adr)


# In[39]:


def reg_eval_metrics(y, ypred):
    r2 = r2_score(y, ypred)
    return r2

r2_score_result = reg_eval_metrics(y_test, ypred_lin_reg)
print("R2_SCORE of Linear regressor", r2_score_result)

r2_score_result = reg_eval_metrics(y_test, ypred_dtree_reg)
print("R2_SCORE of DecisiontreeRegressor", r2_score_result)

r2_score_result = reg_eval_metrics(y_test, ypred_gbr)
print("R2_SCORE of Gradientboost", r2_score_result)

r2_score_result = reg_eval_metrics(y_test, ypred_adr)
print("R2_SCORE of Adaboost", r2_score_result)


# #### Choosing best model as Gradient boosting regressor, which has comparitively high R2 score and less in mean absolute error , mean squared error and rmse. 

# #### I predicted the rating using Gradient boosting regressor model

# In[40]:


df['Predicted_Rating']=gbr.predict(x)
random_df=df.sample(n=20,random_state=42)
random_df


# In[41]:


fig=plt.figure(figsize=(16,6))
fig.suptitle('Relationship between rating and votes')
plt.subplot(1,2,1)
sns.distplot(df[['Rating']]).set_title('Rating of movies')
plt.subplot(1,2,2)
sns.distplot(df[['Predicted_Rating']]).set_title('Predicted ratings of movies')
plt.show()

