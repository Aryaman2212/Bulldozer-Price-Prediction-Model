#!/usr/bin/env python
# coding: utf-8

# # Predicting the sale price of bulldozers using Machine Learning
# 
# * Problem Definition
# * Data
# * Evaluation
# * Features

# ### Importing Tools

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[2]:


df = pd.read_csv(r'D:\OneDrive\Desktop\bluebook-for-bulldozers\bluebook-for-bulldozers\TrainAndValid.csv', low_memory = False)


# In[3]:


df


# In[3]:


df.isna().sum()


# In[5]:


fig, ax = plt.subplots()
ax.scatter(df['saledate'][:1000],df['SalePrice'][:1000])


# In[6]:


df.SalePrice.plot.hist()


# ### Parsing Dates
# 
# * Parsing the SaleDate column since by converting it from an object to a datetime column

# In[4]:


df = pd.read_csv(r'D:\OneDrive\Desktop\bluebook-for-bulldozers\bluebook-for-bulldozers\TrainAndValid.csv', 
                 low_memory = False, 
                 parse_dates= ['saledate'])


# In[8]:


df['saledate'][:1000]


# In[9]:


fig, ax = plt.subplots()
ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000])


# In[10]:


df.head().T


# ### Sort the dataframe by sale date
# 

# In[5]:


df.sort_values(['saledate'],inplace = True, ascending= True)


# In[12]:


df.saledate.head(20)


# In[6]:


df_tmp = df.copy()


# In[14]:


df_tmp.saledate.head(20)


# ### Feature Engineering
# 

# In[7]:


df_tmp['SaleYear'] = df_tmp.saledate.dt.year
df_tmp['SaleDay'] = df_tmp.saledate.dt.day
df_tmp['SaleMonth'] = df_tmp.saledate.dt.month
df_tmp['SaleDayOfWeek'] = df_tmp.saledate.dt.dayofweek
df_tmp['SaleDayOfYear'] = df_tmp.saledate.dt.dayofyear


# In[16]:


df_tmp.state


# In[8]:


df_tmp.drop('saledate',axis = 1, inplace = True)


# In[18]:


df_tmp.state.value_counts()


# In[19]:


pd.api.types.is_string_dtype(df_tmp['UsageBand'])


# In[20]:


df_tmp['UsageBand'].dtype


# In[21]:


pd.api.types.is_object_dtype('UsageBand')


# In[9]:


for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[10]:


for label,content in df_tmp.items():
    if pd.api.types.is_object_dtype(content):
        df_tmp[label] = content.astype('category').cat.as_ordered()


# In[11]:


df_tmp.info()


# In[25]:


df_tmp.state.cat.categories


# In[26]:


df_tmp.state.cat.codes


# ### Missing Values

# In[12]:


df_tmp.isnull().sum()/len(df_tmp)


# ### Save Preprocessed Data

# In[28]:


df_tmp.to_csv(r'D:\OneDrive\Desktop\bluebook-for-bulldozers\bluebook-for-bulldozers\train_tmp.csv', index=False)


# # Numerical Missing Values

# In[29]:


#Checking all numeric columns

for label,content in df_tmp.items():
    
    if pd.api.types.is_numeric_dtype(content):
        print(label)


# In[30]:


#Checking which numeric columns have null values

for label,content in df_tmp.items():
    
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[13]:


# Filling the numeric rows with the median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):  # Check if the column contains numeric data
        if pd.isnull(content).sum():
            # Adding a new binary column which tells us if the data was missing before imputation
            df_tmp[label+'_is_missing'] = pd.isnull(content)
            
            # Imputing the missing numeric values with the median
            df_tmp[label] = content.fillna(content.median())

            


# In[14]:


df_tmp.T


# In[15]:


#Checking if theres still numeric columns with null values

for label,content in df_tmp.items():
    
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[16]:


count =  0

for label,content in df_tmp['auctioneerID_is_missing'].items():
    if content == True:
        count += 1
print(count)


# In[17]:


df_tmp.auctioneerID_is_missing.value_counts()


# # Dealing with Categorical Missing Values

# In[18]:


#Finding columns without numeric data

for label,content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[37]:


df_tmp.to_csv(r'D:\OneDrive\Desktop\bluebook-for-bulldozers\bluebook-for-bulldozers\updated_train_tmp.csv', index=False)


# In[19]:


for label,content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        
        #Add a binary column to check if that value was missing before imputation
        df_tmp[label+'_is_missing'] = pd.isnull(content)
        
        #convert categories into numbers
        
        df_tmp[label] = pd.Categorical(content).codes +1


# In[20]:


df_tmp['UsageBand']


# In[21]:


df_tmp['state']


# In[22]:


#Checking for any leftover missing values
df_tmp.info()


# In[23]:


df_tmp.shape


# In[25]:


df_tmp.head().T


# In[28]:


df_tmp.isna().sum()


# In[32]:


len(df_tmp)


# In[30]:


from sklearn.ensemble import RandomForestRegressor


# In[31]:


get_ipython().run_cell_magic('time', '', '\n#Instantiate Model\n\nmodel = RandomForestRegressor(n_jobs = -1,\n                              random_state = 42)\n\n#Fit the model\n\nmodel.fit(df_tmp.drop("SalePrice",axis = 1),df_tmp[\'SalePrice\'])\n')


# In[33]:


model.score(df_tmp.drop("SalePrice",axis = 1),df_tmp['SalePrice'])


# In[36]:


df_tmp['SaleYear']


# In[37]:


#Splitting Data into training and validation set

df_val = df_tmp[df_tmp.SaleYear == 2012]

df_train = df_tmp[df_tmp.SaleYear != 2012]

len(df_val),len(df_train)


# In[38]:


X_train,y_train = df_train.drop('SalePrice',axis = 1),df_train.SalePrice
X_val,y_val = df_val.drop('SalePrice',axis = 1),df_val.SalePrice

X_train.shape,y_train.shape,X_val.shape,y_val.shape


# In[39]:


y_train


# # Building an Evaluation Function (Root Mean Squared Log Error)

# In[44]:


from sklearn.metrics import mean_squared_log_error, mean_absolute_error,r2_score

def rmsle(y_test,y_preds):
    """
    Calculates Root Mean Squared Log Error between predictions and true errors
    
    """
    return np.sqrt(mean_squared_log_error(y_test,y_preds))

#Creating a function to check our model scores

def model_scores(model):
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    scores = {'Training MAE': mean_absolute_error(y_train,train_preds),
              'Validation MAE': mean_absolute_error(y_val,val_preds),
              'Training RMSLE': rmsle(y_train,train_preds),
              'Validation RMSLE': rmsle(y_val,val_preds),
              'Training R^2 Error': r2_score(y_train,train_preds),
              'Validation R^2 Error': r2_score(y_val,val_preds)}
    
    return scores


# # Testing our model on a subset(To tune the hyperparameters)

# In[41]:


#Change max samples value

model = RandomForestRegressor(n_jobs = -1,
                              random_state = 42,
                              max_samples= 10000)


# In[42]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train,y_train)\n')


# In[45]:


model_scores(model)


# # Hyperparameter tuning with RandomizedSearchCV

# In[46]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.model_selection import RandomizedSearchCV\n\n#Different randomforestregressor hyperparameters\n\nrf_grid = {'n_estimators': np.arange(10,100,10),\n           'max_depth': [None,3,5,10],\n           'min_samples_split': np.arange(2,20,2),\n           'min_samples_leaf': np.arange(1,20,2),\n           'max_features': [0.5,1,'sqrt','auto'],\n           'max_samples': [10000]}\n\n#Instantiate RandomizedSearchCV Model\n\nrs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs = -1,\n                                                    random_state= 42),\n                             param_distributions= rf_grid,\n                             n_iter = 2,\n                             cv = 5,\n                             verbose= True)\n\nrs_model.fit(X_train,y_train)\n")


# In[47]:


rs_model.best_params_


# In[48]:


model_scores(rs_model)


# In[50]:


get_ipython().run_cell_magic('time', '', '\n#Most ideal hyperparameters\n\nideal_model = RandomForestRegressor(n_estimators = 40,\n                                    min_samples_leaf=1,\n                                    min_samples_split=14,\n                                    max_features=0.5,\n                                    n_jobs = -1,\n                                    max_samples = None)\n\nideal_model.fit(X_train,y_train)\n')


# # Assessing our most ideal model

# In[51]:


model_scores(ideal_model)


# # Using our model on the test set

# In[54]:


#Import the test setb

df_test = pd.read_csv(r'D:\OneDrive\Desktop\bluebook-for-bulldozers\bluebook-for-bulldozers\Test.csv', 
                      low_memory = False, 
                      parse_dates = ['saledate'])

df_test.head()


# # Pre-processing the data (Getting the test set in the same format as our training set)

# In[55]:


def preprocess_data(df):
    """
    This function will take in a dataset and return a transformed dataset based on our training data
    
    """
    
    #Feature engineer the date columns
    
    
    df['SaleYear'] = df.saledate.dt.year
    df['SaleDay'] = df.saledate.dt.day
    df['SaleMonth'] = df.saledate.dt.month
    df['SaleDayOfWeek'] = df.saledate.dt.dayofweek
    df['SaleDayOfYear'] = df.saledate.dt.dayofyear
    
    df.drop('saledate',axis = 1, inplace = True)
    
    #Fill the numeric missing rows with the median
    
    # Filling the numeric rows with the median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):  # Check if the column contains numeric data
            if pd.isnull(content).sum():
                # Adding a new binary column which tells us if the data was missing before imputation
                df[label+'_is_missing'] = pd.isnull(content)

                # Imputing the missing numeric values with the median
                df[label] = content.fillna(content.median())

    #Filled the categorical missing data and turned it into numbers
    
        if not pd.api.types.is_numeric_dtype(content):
            
            #Add new columns to mention if the values were missing before imputation
            df[label+'_is_missing'] = pd.isnull(content)
            
            #Convert categories into numbers +1 because pandas converts missing values into -1
            
            df[label] = pd.Categorical(content).codes + 1
    
    
    
    
    return df


# In[56]:


#Process the test data

df_test = preprocess_data(df_test)

df_test.head()


# In[57]:


test_preds = ideal_model.predict(df_test)


# In[58]:


set(X_train.columns) - set(df_test.columns)


# In[59]:


#We will manually add 'auctioneerID_is_missing' column in our test set

df_test['auctioneerID_is_missing'] = False

df_test.head()


# In[60]:


test_preds = ideal_model.predict(df_test)


# In[61]:


#Fixing the order of columns

df_train.columns.values


# In[63]:


df_train.columns.get_loc('SaleDayOfYear')


# In[65]:


df_test.drop('auctioneerID_is_missing',axis = 1, inplace = True)


# In[66]:


df_test.insert(loc=56, column="auctioneerID_is_missing", value=False)


# In[67]:


df_test.shape


# In[68]:


df_test.columns.values


# In[69]:


test_preds = ideal_model.predict(df_test)


# In[70]:


test_preds


# In[71]:


df_preds = pd.DataFrame()

df_preds['SalesID'] = df_test['SalesID']
df_preds['SalesPrice'] = test_preds

df_preds


# In[72]:


#Export Prediction Data

df_preds.to_csv(r'D:\OneDrive\Desktop\bluebook-for-bulldozers\bluebook-for-bulldozers\test_predictions.csv', index=False)


# # Feature Importance

# In[75]:


#Find the feature importance of our best model

ideal_model.feature_importances_


# In[79]:


def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({'Features': columns,
                        'Feature_Importances': importances})
          .sort_values('Feature_Importances', ascending=False)  # Corrected column name
          .reset_index(drop=True))

    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df['Features'][:n], df['Feature_Importances'][:n])
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Features')
    ax.invert_yaxis()  # Invert y-axis for better visualization
    ax.set_title('Top {} Feature Importances'.format(n))
    plt.show()


# In[80]:


plot_features(X_train.columns,ideal_model.feature_importances_)

