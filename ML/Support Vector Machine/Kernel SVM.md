# Kernel SVM
```python
#When we have a data that is non-linearly, a straight line cannot be used as a decision boundary (hence the kernel)

#Implementing Kernel SVM

#Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Reading dataset to pandas dataframe
irisdata=pd.read_csv(url, names=colnames)

#preprocessing the data

#x is the attributes (what we're using to predict)
x=irisdata.drop('Class', axis=1)
#y is the label (what we're trying to predict)
y=irisdata['Class']

#Training Testing Splitting

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

#Training the Algorithm
''' In order to train a kernel SVM we use the same SVC class.
The difference between a kernel SVM and the simple SVM is in the value for the kernel parameter of the SVC class:
In simple SVM we used linear, but here we can use Gaussian, polynomial, sigmoid, or computable kernel'''

#1. Polynomial Kernel
#If you want to use a polynomial kernel, you need to pass an additional value for the degree parameter of the SVC class (The degee of the polynomial)

from sklearn.svm import SVC
svclassifier= SVC(kernel='poly', degree=8)
svclassifier.fit(x_train, y_train)

#Making Predictions
y_predict=svclassifier.predict(x_test)

#Evaluating the Algorithm

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test,y_predict))


#2. Gaussian Kernel
from sklearn.svm import SVC
svclassifier=SVC(kernel='rbf')
svclassifier.fit(x_train, y_train)

#Prediction and Evaluation
y_predict=svclassifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test,y_predict))


#3. Sigmoid Kernel
from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(x_train, y_train)

#Prediction and Evaluation
y_pred = svclassifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Sigmoid function returns two values: 0 and 1, therefore it is better to use it in binary classification problems. 
#Hence the bad score (we had 3 classes) in this example.
```
# Resources
(https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/)

