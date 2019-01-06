K-Fold Cross-Validation - Divide the data into K subgroups where one of them is used for training the model and
the remaining k-1 are training. Then repeat that for each of the k subgroups until you've trained the model on
every subgroup.

```python
#Importing libraries

import pandas as pd
import numpy as np

#Reading the data
path='C:\\Users\\sagi\\Desktop\\Learning\ML\\Datasets\\wineQualityReds.csv'
dataset=pd.read_csv(path) #The dataset was semi-colon separated so we have to pass the ';' attribute to the
# "sep" parameter so pandas is able to properly parse the file.

#Data analysis:
dataset.shape

#Preprocess the data
#Divide into features and labels
X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values

#Remember that we're doing cross validation- we will use all of the data to be in the training set- so we dont need to split it
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=0)   #set the value for the test_size parameter to 0 to make all data be in the training

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#Training
from sklearn.ensemble import RandomForestRegressor
classifier=RandomForestRegressor(n_estimators=200)



#Cross Val
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)
#cross_val_score takes 4 parameters: estimator is the algorithm, second and third are features and labels, and the number of folds is passed to the cv
print(all_accuracies)

#Output:
#[0.81681873 0.76539769 0.77995326 0.73214267 0.78790996]
print(all_accuracies.mean()) #avg all accuracies

print(all_accuracies.std())
#Output:
#0.027= 2.7% which is good. low variance means the model will perform more or less similar on all test sets.

# Resources
https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/
