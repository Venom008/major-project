#!/usr/bin/env python
# coding: utf-8

# In[194]:


#Ashay Priyanka Aditya Chandan#


# In[195]:


import pandas as pd
import numpy as np


# In[196]:


data = pd.read_csv("D:\Major project Data\Airfoil_Noise.csv")


# In[197]:


data.head()


# In[198]:


y=data['Scaled sound pressure level, in decibels.']
data.drop(['Scaled sound pressure level, in decibels.'],axis=1, inplace=True)
X=data


# In[199]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[200]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model = model.fit(X_train, y_train)


# In[201]:


y_pred = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
r2 = r2_score(y_train, y_pred)
mae = mean_absolute_error(y_train,y_pred)
print ("For Training Dataset")
print ("Mean absolute error: " , mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)
y_pred = model.predict (X_test)
rmse = np.sqrt (mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print ("For Testing Dataset")
print ("Mean absolute error: ", mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)


# In[202]:


import matplotlib.pyplot as plt
plt.scatter (y_test, y_pred, c='g')
plt.plot (y_test, y_test, c='red' )
plt. show()


# In[203]:


#Ridge Regularization
#Ashay
#Aditya 
#Priyanka
#Chandan


# In[204]:


from sklearn.linear_model import Ridge
model = Ridge()
model = model.fit(X_train, y_train)


# In[205]:


y_pred = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
r2 = r2_score(y_train, y_pred)
mae = mean_absolute_error(y_train,y_pred)
print ("For Training Dataset")
print ("Mean absolute error: " , mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)
y_pred = model.predict (X_test)
rmse = np.sqrt (mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print ("For Testing Dataset")
print ("Mean absolute error: ", mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)


# In[206]:


import matplotlib.pyplot as plt
plt.scatter (y_test, y_pred, c='g')
plt.plot (y_test, y_test, c='red' )
plt. show()


# In[207]:


#Lasso Regularization
#Ashay
#Aditya 
#Priyanka
#Chandan


# In[208]:


from sklearn.linear_model import Lasso
model = Lasso()
model = model.fit(X_train, y_train)


# In[209]:


y_pred = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
r2 = r2_score(y_train, y_pred)
mae = mean_absolute_error(y_train,y_pred)
print ("For Training Dataset")
print ("Mean absolute error: " , mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)
y_pred = model.predict (X_test)
rmse = np.sqrt (mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print ("For Testing Dataset")
print ("Mean absolute error: ", mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)


# In[210]:


import matplotlib.pyplot as plt
plt.scatter (y_test, y_pred, c='g')
plt.plot (y_test, y_test, c='red' )
plt. show()


# In[211]:


#Elastic Net Regularization
#Ashay
#Aditya 
#Priyanka
#Chandan


# In[212]:


from sklearn.linear_model import ElasticNet
model = ElasticNet()
model = model.fit(X_train, y_train)


# In[213]:


y_pred = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
r2 = r2_score(y_train, y_pred)
mae = mean_absolute_error(y_train,y_pred)
print ("For Training Dataset")
print ("Mean absolute error: " , mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)
y_pred = model.predict (X_test)
rmse = np.sqrt (mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print ("For Testing Dataset")
print ("Mean absolute error: ", mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)


# In[214]:


import matplotlib.pyplot as plt
plt.scatter (y_test, y_pred, c='g')
plt.plot (y_test, y_test, c='red' )
plt. show()


# In[215]:


#Polynomial Regression
#Ashay
#Aditya 
#Priyanka
#Chandan


# In[216]:


from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=2)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.fit_transform(X_test)
model = LinearRegression()
model = model.fit(X_train_poly, y_train)


# In[217]:


y_pred = model.predict(X_train_poly)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
r2 = r2_score(y_train, y_pred)
mae = mean_absolute_error(y_train,y_pred)
print ("For Training Dataset")
print ("Mean absolute error: " , mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)
y_pred = model.predict (X_test_poly)
rmse = np.sqrt (mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print ("For Testing Dataset")
print ("Mean absolute error: ", mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)


# In[218]:


import matplotlib.pyplot as plt
plt.scatter (y_test, y_pred, c='g')
plt.plot (y_test, y_test, c='red' )
plt. show()


# In[219]:


from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=3)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.fit_transform(X_test)
model = LinearRegression()
model = model.fit(X_train_poly, y_train)


# In[220]:


y_pred = model.predict(X_train_poly)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
r2 = r2_score(y_train, y_pred)
mae = mean_absolute_error(y_train,y_pred)
print ("For Training Dataset")
print ("Mean absolute error: " , mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)
y_pred = model.predict (X_test_poly)
rmse = np.sqrt (mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print ("For Testing Dataset")
print ("Mean absolute error: ", mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)


# In[221]:


import matplotlib.pyplot as plt
plt.scatter (y_test, y_pred, c='g')
plt.plot (y_test, y_test, c='red' )
plt. show()


# In[222]:


from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=4)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.fit_transform(X_test)
model = LinearRegression()
model = model.fit(X_train_poly, y_train)


# In[223]:


y_pred = model.predict(X_train_poly)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
r2 = r2_score(y_train, y_pred)
mae = mean_absolute_error(y_train,y_pred)
print ("For Training Dataset")
print ("Mean absolute error: " , mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)
y_pred = model.predict (X_test_poly)
rmse = np.sqrt (mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print ("For Testing Dataset")
print ("Mean absolute error: ", mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)


# In[224]:


import matplotlib.pyplot as plt
plt.scatter (y_test, y_pred, c='g')
plt.plot (y_test, y_test, c='red' )
plt. show()


# In[225]:


# Decision Tree Regression
#Ashay
#Aditya 
#Priyanka
#Chandan


# In[226]:


from sklearn.tree import DecisionTreeRegressor
regr = DecisionTreeRegressor(random_state=0)
regr = regr.fit(X_train, y_train)


# In[227]:


y_pred = regr.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
r2 = r2_score(y_train, y_pred)
mae = mean_absolute_error(y_train,y_pred)
print ("For Training Dataset")
print ("Mean absolute error: " , mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)
y_pred = regr.predict (X_test)
rmse = np.sqrt (mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print ("For Testing Dataset")
print ("Mean absolute error: ", mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)


# In[228]:


import matplotlib.pyplot as plt
plt.scatter (y_test, y_pred, c='g')
plt.plot (y_test, y_test, c='red' )
plt. show()


# In[229]:


#Random Forest Regression
#Ashay
#Aditya 
#Priyanka
#Chandan


# In[230]:


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor (max_depth=3, random_state=0, n_estimators=100)
regr = regr.fit(X_train, y_train)


# In[231]:


y_pred = regr.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
r2 = r2_score(y_train, y_pred)
mae = mean_absolute_error(y_train,y_pred)
print ("For Training Dataset")
print ("Mean absolute error: " , mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)
y_pred = regr.predict (X_test)
rmse = np.sqrt (mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print ("For Testing Dataset")
print ("Mean absolute error: ", mae)
print ("Root mean square error: ", rmse)
print ("R2 score: ", r2)


# In[232]:


import matplotlib.pyplot as plt
plt.scatter (y_test, y_pred, c='g')
plt.plot (y_test, y_test, c='red' )
plt. show()


# In[ ]:





# In[ ]:




