```python
#How the Random Forest Algorithm Works
'''
"1.Pick N random records from the dataset.
2.Build a decision tree based on these N records.
3.Choose the number of trees you want in your algorithm and repeat steps 1 and 2.
4.In case of a regression problem, for a new record, each tree in the forest predicts a value for Y (output).
The final value can be calculated by taking the average of all the values predicted by all the trees in forest.
Or, in case of a classification problem, each tree in the forest predicts the category to which the new record belongs.
Finally, the new record is assigned to the category that wins the majority vote."
'''
"The random forest algorithm works well when you have both categorical and numerical features."
"The random forest algorithm also works well when data has missing values or it has not been scaled well. "

#Using Random Forest for Regression

#predict the gas consumption (in millions of gallons) in 48 of the US states based on petrol tax (in cents), per capita
#income (dollars), paved highways (in miles) and the proportion of population with the driving license.

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the dataset
path='C:\\Users\\sagi\\Desktop\\Learning\\ML\\Datasets\\petrol_consumption.csv'
dataset=pd.read_csv(path)


#Training, Splitting, Testing
x=dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Scaling

#We need to scale our data because Average_Income field has values in the range of thousands while Petrol_tax has values in range of tens (for example)
#Though, it's not too important for the random tree algorithm.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Training

from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor(n_estimators=200)
#The n_estimators parameter defines the number of trees in the random forest
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


#Evaluating
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#As an example, the root mean squared error is 64.93 which is greater than 10 percent of the average petrol consumption
#i.e. 576.77. Indicating that we have not used enough estimators (trees).
```
# Reasources
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
