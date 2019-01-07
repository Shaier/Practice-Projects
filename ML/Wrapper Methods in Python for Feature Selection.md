# Applying Wrapper Methods in Python for Feature Selection
When you want to choose a specific ML algorithm to train your model features selection with filter may not be the best.
Wrapper methods select the most optimal features for the specified algorithm
"Wrapper methods are based on greedy search algorithms as they evaluate all possible combinations of the features and 
select the combination that produces the best result for a specific machine learning algorithm."
There are 3 categories of these methods:
"Step forward feature selection, Step backwards feature selection and Exhaustive feature selection"

```python

#Step Forward Feature Selection
'''
1st step: performance of the classifier is evaluated with respect to each feature. The feature that performs the best is selected
2nd: the first feature is tried in combination with all the other features and the best combination is selected. 
Contiue until the specified number of features are selected
'''
#We need to convert categorical into numerical values to perform Step forward. But we will just remove them here.

#Preprocess (Libraries, data, select for numerical only, train,split, remove correlated features, check for shape)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

paribas_data = pd.read_csv(r"E:\Datasets\paribas_data.csv", nrows=20000)
paribas_data.shape

num_colums = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(paribas_data.select_dtypes(include=num_colums).columns)
paribas_data = paribas_data[numerical_columns]
paribas_data.shape

train_features, test_features, train_labels, test_labels = train_test_split(
    paribas_data.drop(labels=['target', 'ID'], axis=1),
    paribas_data['target'],
    test_size=0.2,
    random_state=41)

correlated_features = set()
correlation_matrix = paribas_data.corr()
for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)


train_features.drop(labels=correlated_features, axis=1, inplace=True)
test_features.drop(labels=correlated_features, axis=1, inplace=True)

train_features.shape, test_features.shape


#Implementing Step Forward Feature Selection
#SequentialFeatureSelector will help us select for the most optimal features
#Random Forest Classifier will help us select for the most optimal parameters.
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from mlxtend.feature_selection import SequentialFeatureSelector

feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=15, #selecting the 15 features that performed the best
           forward=True,
           verbose=2, #logging the progress of the feature selector
           scoring='roc_auc', #defines the performance evaluation criteria
           cv=4) #cross-validation folds.



#Train
features = feature_selector.fit(np.array(train_features.fillna(0)), train_labels)


#Seeing the 15 selected features
filtered_features= train_features.columns[list(features.k_feature_idx_)]
filtered_features

#Seeing the classification performance of the random forest algorithm using these 15 features
clf = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=3)
clf.fit(train_features[filtered_features].fillna(0), train_labels)

train_pred = clf.predict_proba(train_features[filtered_features].fillna(0))
print('Accuracy on training set: {}'.format(roc_auc_score(train_labels, train_pred[:,1])))

test_pred = clf.predict_proba(test_features[filtered_features].fillna(0))
print('Accuracy on test set: {}'.format(roc_auc_score(test_labels, test_pred [:,1])))

#Since the accuracy on both the training and testing is similar our model is not overfitting

############################################################

#Step Backwards Feature Selection
'''
"1st step: one feature is removed at a time and the performance of the classifier is evaluated. The feature set that yields the best performance is retained
2nd step: one feature is removed in again and the performance of all the combination of features except the 2 features is evaluated
continues until the specified number of features remain in the dataset
"
'''

#Step Backwards Feature Selection

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector

feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=15,
           forward=False, #Notice the false - implies backwards
           verbose=2,
           scoring='roc_auc',
           cv=4)

features = feature_selector.fit(np.array(train_features.fillna(0)), train_labels)

#Seeing the feature selected
filtered_features= train_features.columns[list(features.k_feature_idx_)]
filtered_features

#Evaluating the performance of the random forest classifier on the features selected
clf = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=3)
clf.fit(train_features[filtered_features].fillna(0), train_labels)

train_pred = clf.predict_proba(train_features[filtered_features].fillna(0))
print('Accuracy on training set: {}'.format(roc_auc_score(train_labels, train_pred[:,1])))

test_pred = clf.predict_proba(test_features[filtered_features].fillna(0))
print('Accuracy on test set: {}'.format(roc_auc_score(test_labels, test_pred [:,1])))

##################################################

#Exhaustive Feature Selection

'''
"The performance of a machine learning algorithm is evaluated against all possible combinations of the features in the dataset
The feature subset that yields best performance is selected
The exhaustive search algorithm try all the combination of features and selects the best.
"
'''

#Exhaustive Feature Selection
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

feature_selector = ExhaustiveFeatureSelector(RandomForestClassifier(n_jobs=-1),
           min_features=2,
           max_features=4,
           scoring='roc_auc',
           print_progress=True,
           cv=2)


#Pass training, testing data to the selector
features = feature_selector.fit(np.array(train_features.fillna(0)), train_labels)

#Seeing the performance of random forest classifier on the features selected as a result of exhaustive feature selection
clf = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=3)
clf.fit(train_features[filtered_features].fillna(0), train_labels)

train_pred = clf.predict_proba(train_features[filtered_features].fillna(0))
print('Accuracy on training set: {}'.format(roc_auc_score(train_labels, train_pred[:,1])))

test_pred = clf.predict_proba(test_features[filtered_features].fillna(0))
print('Accuracy on test set: {}'.format(roc_auc_score(test_labels, test_pred [:,1])))

```
# Resources
https://stackabuse.com/applying-wrapper-methods-in-python-for-feature-selection/
