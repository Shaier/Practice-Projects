# K-Nearest Neighbors with Scikit-Learn
```python 
"The KNN doesn't have a specialized training phase. Rather, it uses all of the data for training while classifying a new data point or instance.
KNN is a non-parametric learning algorithm, which means that it doesn't assume anything about the underlying data.
most of the real world data doesn't really follow any theoretical assumption e.g. linear-separability, uniform distribution, etc."
For pros and cons on using this visit the website in the resources

#The general idea: KNN calculates the distance of a new data point to all other data points, and then
#it chooses the K-nearest data points (K can be any integer) and assigns that data point to the class to which the majority of the K data points belong.

#Loading the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url= 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# Assign col names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

#read the dataset
dataset=pd.read_csv(url, names=names)

#Preprocessing
#splitting the data into x(attributes) and y(label)
#The X variable contains the first four columns of the dataset
#y contains the labels.

x=dataset.iloc[:, :-1].values #the iloc takes [row, col] --> so all rows and only 4 col.
y=dataset.iloc[:, 4].values #just the 5th col.


#Train Test Split
#In order to prevent overfitting we need to divide the data into training and testing as usual.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

#Feature Scaling
"Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated."
#Explanation by Wiki:
'''Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will
not work properly without normalization. For example, the majority of classifiers calculate the distance between two 
points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by 
this particular feature. Therefore, the range of all features should be normalized so that each feature contributes
approximately proportionately to the final distance.
'''

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit (x_train)

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#Training and Predictions

from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5) #the value for k here is 5
#K doesnt have an ideal value. It is selected after testing and evaluation
classifier.fit(x_train,y_train)

#make prediction
y_predict=classifier.predict(x_test)

#Evaluating the Algorithm

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

#Comparing Error Rate with the K Value
#Let's try changing K to see if we get better results
"One way to help you find the best value of K is to plot the graph of K value and the corresponding error rate for the dataset."
#plotting the mean error for the predicted values of test set for all the K values between 1 and 40.

#calculating the mean of error for all of the predicted values (K: 1-40)
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

#In each itiration the mean error for predicted values of test set is calculated and added to the 'error' list

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

#Plotting this we see that the mean error is 0 when K is ranging between 5 and 18
```
# Resources
https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

