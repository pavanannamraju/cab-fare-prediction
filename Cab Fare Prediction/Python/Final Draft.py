#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries required

# In[121]:


from numpy import where
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Reading the Datasets

# In[122]:


train = pd.read_csv('../datasets/train_cab.csv')


# In[123]:


test = pd.read_csv('../datasets/test.csv')


# In[124]:


train.head()


# In[125]:


test.head()


# # Preprocessing Dataset

# In[126]:


train.info()


# In[127]:


from pprint import pprint


# In[128]:


pprint(test.info())


# `Missing Values` --> fare_amount, passenger_count <br>
# `Type Casting`   --> fare_amount, pickup_datetime

# In[129]:


test.info()


# `Type Casting`   --> fare_amount, pickup_datetime

# ## Dealing with missing values in Train set

# In[130]:


train.fare_amount.isnull().sum()


# As fare_amount would be the target variable, it's always better to remove those missing rows.

# In[131]:


train.dropna(subset=['fare_amount'], inplace=True)


# In[132]:


train.fare_amount.isnull().sum()


# In[133]:


train.passenger_count.isnull().sum()


# In[134]:


55/(len(train))*100


# <b>The missing values in passenger count is less than 0.5 percent. So dropping the values</b>

# In[135]:


train.dropna(subset=['passenger_count'], inplace=True)


# In[136]:


train.shape


# * All missing values are handled

# ## Handling Type Conversion in both training and testing data

# In[137]:


train.info()


# In[138]:


#train.fare_amount.astype(float) # Throws an error ValueError: could not convert string to float: '430-'


# In[139]:


train[train.fare_amount == '430-']


# In[140]:


train = train[~(train.fare_amount == '430-')]


# In[141]:


train.shape


# In[142]:


train.pickup_datetime[1327]


# In[143]:


train.drop(1327,inplace=True)


# In[144]:


train.fare_amount = train.fare_amount.astype(float)


# In[145]:


train.info()


# ## Feature Engineering for both training and testing data

# In[146]:


def split_dates(df):
    df.pickup_datetime = df.pickup_datetime.apply(lambda x: x[:-3])
    df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
    df['Year'] = df.pickup_datetime.dt.year
    df['Month'] = df.pickup_datetime.dt.month
    df['Day'] = df.pickup_datetime.dt.day
    df['Hour'] = df.pickup_datetime.dt.hour
    df['Minutes'] = df.pickup_datetime.dt.minute
    df.drop('pickup_datetime', axis=1, inplace=True)

    return df


# In[147]:


train = split_dates(train)


# In[148]:


test = split_dates(test)


# In[149]:


train.head()


# In[150]:


test.head()


# In[151]:


train.reset_index(drop=True, inplace=True)


# In[152]:


test.reset_index(drop=True, inplace=True)


# In[153]:


import haversine


# In[154]:


out = []
for i,j,k,l in zip(*(train.pickup_latitude, train.pickup_longitude, train.dropoff_latitude, train.dropoff_longitude)):
    out.append(haversine.haversine((i,j),(k,l)))

train['distance'] = pd.Series(out)

train.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude'], axis=1, inplace=True)


# In[155]:


out = []
for i,j,k,l in zip(*(test.pickup_latitude, test.pickup_longitude, test.dropoff_latitude, test.dropoff_longitude)):
    out.append(haversine.haversine((i,j),(k,l)))

test['distance'] = pd.Series(out)

test.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude'], axis=1, inplace=True)


# In[156]:


train.head()


# In[157]:


test.head()


# # Sanity Check

# Sanity checks to be performed<br>
# 
# - fare amount should not be less than or equal to zero
# - We also have to convert the distance of the ride using pickup and dropoff coordinates
# - Then we have to check for the distances less than or equal to 0
# - Check Passenger count, which should be integer and less than or equal to 6 and greater than 0

# In[158]:


train[train.fare_amount<=0]


# In[159]:


train = train[train.fare_amount>0]


# In[160]:


train[train.distance <= 0]


# In[161]:


train = train[~(train.distance <= 0)]


# In[162]:


test = test[~(test.distance <= 0)]


# In[163]:


train = train[train.passenger_count > 0]


# In[164]:


train = train[train.passenger_count<=6]


# In[165]:


train.shape


# In[166]:


train.passenger_count.value_counts()


# In[167]:


train = train[~(train.passenger_count.isin([0.12,1.3]))]


# In[168]:


train.passenger_count = train.passenger_count.astype(int)


# In[169]:


train.reset_index(drop=True, inplace=True)


# In[170]:


test.reset_index(drop=True, inplace=True)


# # outlier check

# In[171]:


import seaborn as sns
from scipy.stats import probplot


# In[172]:


def outlier_check_graphs(df, column):
    plt.figure(figsize=[20,6])
    plt.subplot(1,3,1)
    sns.distplot(df[column])
    
    plt.subplot(1,3,2)
    probplot(df[column], dist='norm', plot=plt)
    
    plt.subplot(1,3,3)
    sns.boxplot(df[column])
    plt.show()
    


# In[173]:


train.fare_amount.describe()


# In[174]:


outlier_check_graphs(train, 'fare_amount')


# In[175]:


def find_limits(df, column):
    q25 = df[column].quantile(0.25)
    q75 = df[column].quantile(0.75)
    IQR = q75 - q25
    
    lower_limit =  q25 - (IQR * 1.5)
    upper_limit = q75 + (IQR * 1.5)
    return lower_limit, upper_limit


# In[176]:


fare_lower, fare_upper = find_limits(train, 'fare_amount')


# In[177]:


fare_upper


# In[178]:


fare_lower


# In[179]:


from numpy import where


# In[180]:


train['fare_amount']= where(train['fare_amount'] > fare_upper, fare_upper,
                       where(train['fare_amount'] < fare_lower, fare_lower, train['fare_amount']))


# In[181]:


outlier_check_graphs(train,'fare_amount')


# In[182]:


outlier_check_graphs(train, 'distance')


# In[183]:


dist_lower, dist_upper = find_limits(train, 'distance')


# In[184]:


train['distance'] = where(train['distance']> dist_upper, dist_upper,
                         where(train['distance']<dist_lower, dist_lower, train['distance']))


# In[185]:


outlier_check_graphs(train, 'distance')


# # Exploratory Data Analysis
# 
# <b>Goal of EDA</b>: Find relationships between all independant variables to dependant variable

# In[186]:


train.columns


# `Categorical Variables` --> Passenger Count, Year, Month, Day, Hour
# 
# `Numeric Variables`     --> Fare Amount, Distance

# In[187]:


plt.figure(figsize=(10,8))
sns.scatterplot(train.fare_amount, train.distance, color='green')

plt.title('Distance Vs Fare Amount', fontdict={
    'fontsize':18,
    'color':'darkred'
})

plt.xlabel('Fare Amount', fontdict={
    'fontsize':15,
    'color':'blue',
    'fontweight':12
})

plt.ylabel('Distance', fontdict={
    'fontsize':15,
    'color':'blue',
    'fontweight':12
})


# In[188]:


plt.figure(figsize=(15,7))
sns.violinplot(train.Year, train.fare_amount)

plt.title('Year Vs Fare Amount', fontdict={
    'fontsize':18,
    'color':'darkred'
})

plt.xlabel('Fare Amount', fontdict={
    'fontsize':15,
    'color':'blue',
    'fontweight':12
})

plt.ylabel('Year', fontdict={
    'fontsize':15,
    'color':'blue',
    'fontweight':12
})


# In[189]:


plt.figure(figsize=(15,7))
sns.violinplot(train.Month, train.fare_amount)

plt.title('Month Vs Fare Amount', fontdict={
    'fontsize':18,
    'color':'darkred'
})

plt.xlabel('Fare Amount', fontdict={
    'fontsize':15,
    'color':'blue',
    'fontweight':12
})

plt.ylabel('Month', fontdict={
    'fontsize':15,
    'color':'blue',
    'fontweight':12
})


# In[190]:


plt.figure(figsize=(18,7))
sns.violinplot(train.Day, train.fare_amount)

plt.title('Day Vs Fare Amount', fontdict={
    'fontsize':18,
    'color':'darkred'
})

plt.xlabel('Fare Amount', fontdict={
    'fontsize':15,
    'color':'blue',
    'fontweight':12
})

plt.ylabel('Day', fontdict={
    'fontsize':15,
    'color':'blue',
    'fontweight':12
})


# In[191]:


plt.figure(figsize=(18,7))
sns.violinplot(train.passenger_count, train.fare_amount)

plt.title('Passenger Count Vs Fare Amount', fontdict={
    'fontsize':18,
    'color':'darkred'
})

plt.xlabel('Fare Amount', fontdict={
    'fontsize':15,
    'color':'blue',
    'fontweight':12
})

plt.ylabel('Passenger Count', fontdict={
    'fontsize':15,
    'color':'blue',
    'fontweight':12
})


# In[192]:


sns.heatmap(train.corr())


# # Training the data

# Clearly the problem is a regression problem. So the models that I am going to use are:
# 
# - Linear Regression
# - Decision Tree Regression
# - Ridge Regression
# - Lasso Regression
# - Support Vector Regression
# - Random Forest Regression
# - Gradient Boosting Regressor

# Splitting the data into training and validation sets.

# In[193]:


from sklearn.model_selection import train_test_split


# In[194]:


X = train.drop('fare_amount', axis=1)
y = train['fare_amount']


# In[195]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[196]:


from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ## Importing Metrics

# In[197]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ## Defining a function to calculate score

# In[198]:


def calc_scores(X, y, estimator):
    print('R squared',r2_score(y, estimator.predict(X)))
    print('Mean Abosulte Error', mean_absolute_error(y, estimator.predict(X)))
    print('Mean Squared Error',mean_squared_error(y, estimator.predict(X))) 
    print(' ')


# ## Linear Regression

# In[199]:


lr = LinearRegression()
lr.fit(X_train,y_train)
calc_scores(X_train, y_train, lr)
calc_scores(X_test, y_test, lr)


# ## Decision Tree Regression

# In[200]:


dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
calc_scores(X_train, y_train, dtr)
calc_scores(X_test, y_test, dtr)


# ## Lasso Regression

# In[201]:


lasso = Lasso()
lasso.fit(X_train, y_train)
calc_scores(X_train, y_train, lasso)
calc_scores(X_test, y_test, lasso)


# # Support Vector Regression

# In[202]:


svr = SVR()
svr.fit(X_train, y_train)
calc_scores(X_train, y_train, svr)
calc_scores(X_test, y_test, svr)


# ## Random Forest Regression

# In[203]:


rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
calc_scores(X_train, y_train, rfr)
calc_scores(X_test, y_test, rfr)


# ## Gradient Boosting Regression

# In[204]:


gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
calc_scores(X_train, y_train, gbr)
calc_scores(X_test, y_test, gbr)


# Obviously, the best models from above are: `Gradient Boosting`, `Random Forest`

# # Predicting test values using `Random Forest Regressor` and `Gradient Boosting Regression`

# In[205]:


rfr = RandomForestRegressor()


# In[206]:


rfr.fit(X,y)


# In[207]:


rfr.score(X,y)


# In[208]:


test['fare_amount'] = 0


# In[209]:


test['fare_amount'] = rfr.predict(test.drop('fare_amount', axis=1))


# In[210]:


gbr = GradientBoostingRegressor()


# In[211]:


gbr.fit(X,y)


# In[212]:


gbr.score(X,y)


# In[213]:


test1 = test.drop('fare_amount', axis=1)


# In[214]:


test1['fare_amount'] = gbr.predict(test.drop('fare_amount', axis=1))


# ### writing the predictions to a new file

# In[215]:


test.to_csv('final_predictions_python.csv')


# In[ ]:




