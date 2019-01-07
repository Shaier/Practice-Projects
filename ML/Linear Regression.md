# Linear Regression in Python with Scikit-Learn

```python
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='C:\\Users\\sagi\\Desktop\\Learning\\ML\\Datasets\\student_scores.csv'
dataset = pd.read_csv(path)

#Explore the data
dataset.shape
dataset.describe()
dataset.head()

dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

#Preprocessing the data
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values #We're trying to predict y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Training
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

#Getting the intercept
print(regressor.intercept_)

#Getting the slope
print(regressor.coef_)

#Making prediction and testing for accuracy
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#RMSE is less than 10% which is good
```

# Multiple Linear Regression
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='C:\\Users\\sagi\\Desktop\\Learning\\ML\\Datasets\\petrol_consumption.csv'
dataset = pd.read_csv(path)

#Explore the data
dataset.shape
dataset.describe()
dataset.head()

#Preprocess the data
X = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways',
       'Population_Driver_licence(%)']]
y = dataset['Petrol_Consumption']

#Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Train
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

#Making prediction
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

#Evaluation
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#RMSE is slightly greater than 10% of the mean value of the gas consumption in all states
#So the model is not very accurate, but it can still be useful to make predictions
```
# Resources
https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
