#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd

train_data = pd.read_csv('../data/house-prices/train.csv')
test_data = pd.read_csv('../data/house-prices/test.csv')
train_data.head()


# In[7]:


train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']].isnull().sum() 


# In[8]:


features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = train_data[features]
y = train_data['SalePrice']


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[11]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[12]:


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

