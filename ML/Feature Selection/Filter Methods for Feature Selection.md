# Applying Filter Methods in Python for Feature Selection
```python
"Unnecessary and redundant features not only slow down the training time of an algorithm, but they also affect the " \
"performance of the algorithm. The process of selecting the most suitable features for training the machine learning model is called feature selection"

#Removing Constant features
"Constant features are the type of features that contain only one value for all the outputs in the dataset"

#Importing Libraries+ reading file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

santandar_data = pd.read_csv(path, nrows=40000) #taking only 40,000 rows

#Explore the data
santandar_data.shape

#Splitting Data
train_features, test_features, train_labels, test_labels=train_test_split(
    santandar_data.drop(labels=['TARGET'], axis=1),
    santandar_data['TARGET'],
    test_size=0.2,
    random_state=41)

#Removing Constant Features using Variance Threshold
#passing a zero to VarianceThreshold for the parameter will filter all the features with zero variance
constant_filter = VarianceThreshold(threshold=0)

#Apply to training data
constant_filter.fit(train_features)

#Seeing the number of features that are not constant
len(train_features.columns[constant_filter.get_support()]) #using get_support() method of the filter that we created

#Removing constant features from training and test data
#Using the transform() method of the constant_filter
train_features = constant_filter.transform(train_features)
test_features = constant_filter.transform(test_features)

train_features.shape, test_features.shape


#################################

#Removing Quasi-Constant features
#Quasi-constant features are almost constants (same values for a very large subset of the outputs)
#This time we will pass 0.01 to the threshold parameter. Meaning that if the variance of the values in a column is less than 0.01 it will be filtered out
#(removing feature column where approximately 99% of the values are similar)

#Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

santandar_data = pd.read_csv(r"E:\Datasets\santandar_data.csv", nrows=40000)
santandar_data.shape


#Splitting
train_features, test_features, train_labels, test_labels = train_test_split(
    santandar_data.drop(labels=['TARGET'], axis=1),
    santandar_data['TARGET'],
    test_size=0.2,
    random_state=41)

#Removing Constant Features using Variance Threshold
#Removing constant features first
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(train_features)

len(train_features.columns[constant_filter.get_support()])

constant_columns = [column for column in train_features.columns
                    if column not in train_features.columns[constant_filter.get_support()]]

train_features.drop(labels=constant_columns, axis=1, inplace=True)
test_features.drop(labels=constant_columns, axis=1, inplace=True)

#Removing Quasi-Constant
qconstant_filter = VarianceThreshold(threshold=0.01)

#Training
qconstant_filter.fit(train_features)

#Checking the number of non-Quasi
len(train_features.columns[qconstant_filter.get_support()])

#Printing the names of the quasi constants
qconstant_columns = [column for column in train_features.columns
                    if column not in train_features.columns[qconstant_filter.get_support()]]

for column in qconstant_columns:
    print(column)


#Lastly... Seeing if our training and test sets only contains the non-constant and non-quasi-constant columns
train_features = qconstant_filter.transform(train_features)
test_features = qconstant_filter.transform(test_features)

train_features.shape, test_features.shape


#############################################################

#Removing Duplicate Features
#Duplicate features are features that have similar values

#Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

santandar_data = pd.read_csv(r"E:\Datasets\santandar_data.csv", nrows=20000)
#we only take the first 20,000 rows becuase we have to take the transpose of the data matrix before we can remove duplicate features - which is comp. expensive
santandar_data.shape

#Splitting
train_features, test_features, train_labels, test_labels = train_test_split(
    santandar_data.drop(labels=['TARGET'], axis=1),
    santandar_data['TARGET'],
    test_size=0.2,
    random_state=41)

#Removing Duplicate Features using Transpose
#Transposing
train_features_T = train_features.T
train_features_T.shape

#duplicated() method can find duplicate rows from the dataframe (the rows of the transposed dataframe are actually the columns)
print(train_features_T.duplicated().sum()) #total number of duplicate using sum()

#dropping the duplicate
#remove all the duplicate rows and will take transpose of the transposed training set to get the original dataset
unique_features = train_features_T.drop_duplicates(keep='first').T #all the duplicate rows will be dropped except the first copy

#shape of our new training set
unique_features.shape

#names of the duplicate columns
duplicated_features = [dup_col for dup_col in train_features.columns if dup_col not in unique_features.columns]
duplicated_features

#######################################
#Removing Correlated Features
"A dataset can also contain correlated features- Two or more than two features are correlated if they are close to each other in the linear space"
"the weight of the fruit basket is normally correlated with the price. The more the weight, the higher the price"
"If two or more than two features are mutually correlated, they convey redundant information"

#Libraries+ data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

paribas_data = pd.read_csv(r"E:\Datasets\paribas_data.csv", nrows=20000) #20,000 rows and 133 features
paribas_data.shape

#In order to find the correlation we need to use only the features that has numerical values- so we'll filter out the ones that don't

#Preprocessing data
#Removing non numeric features
num_colums = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] #A list that have the datatypes of the columns that we want to retain in our dataset (only numerical)
numerical_columns = list(paribas_data.select_dtypes(include=num_colums).columns) #select_dtypes() method returns the names of the specified numeric columns (we store them in numerical_columns)
paribas_data = paribas_data[numerical_columns] #filter our columns from the dataset

paribas_data.shape

#Splitting
train_features, test_features, train_labels, test_labels = train_test_split(
    paribas_data.drop(labels=['target', 'ID'], axis=1),
    paribas_data['target'],
    test_size=0.2,
    random_state=41)

#Removing Correlated Features using corr() Method
"corr() method returns a correlation matrix containing correlation between all the columns of the dataframe"
"We can then loop through the correlation matrix and see if the correlation between two columns is greater than " \
"threshold correlation, add that column to the set of correlated columns. We can remove that set of columns from the actual dataset."

#Creating correlation matrix
correlated_features = set() #names of all the correlated features.
correlation_matrix = paribas_data.corr() # all the columns in our dataset


#Then we loop through all of the columns in the correlation_matrix and add the columns with a correlation value of 0.8 to the correlated_features set
for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)


#number of columns in the dataset with correlation value greater than 0.8
len(correlated_features)
print(correlated_features)


#removes these columns
train_features.drop(labels=correlated_features, axis=1, inplace=True)
test_features.drop(labels=correlated_features, axis=1, inplace=True)
```
# Resources
https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/







