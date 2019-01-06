"Decision trees can be used to predict both continuous and discrete values i.e. they work well for both regression and classification tasks."

# Decision Tree for Classification

```python
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
path='C:\\Users\\sagi\\Desktop\\Learning\\ML\\Datasets\\bill_authentication.csv'
dataset = pd.read_csv(path)

#Data Analysis
dataset.shape #"(1372,5)" --> 1372 records and 5 attributes.

#Preprocessing the data
X = dataset.drop('Class', axis=1)
y = dataset['Class']

#Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Training
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#After the classifier has been train we can make predictions
y_pred = classifier.predict(X_test)

#Evaluating
#After training the classifier and making predictions we should check how accurate the classifier is

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Out of 275 test instances the algorithm misclassified only 4.
```

# Decision Tree for Regression
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='C:\\Users\\sagi\\Desktop\\Learning\\ML\\Datasets\\petrol_consumption.csv'
dataset = pd.read_csv(path)

#To see statistical details of the dataset
dataset.describe()

X = dataset.drop('Petrol_Consumption', axis=1)
y = dataset['Petrol_Consumption']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Training
#Note the change of class from the earlier example
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#compare some of our predicted values with the actual values and see the accuracy
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#mean absolute error is 47.7, which is less than 10 percent of the mean of all the values in the 'Petrol_Consumption'
# column. This means that our algorithm did a fine prediction job

# Reasources
https://stackabuse.com/decision-trees-in-python-with-scikit-learn/

