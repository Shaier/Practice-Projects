```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Importing the dataset
df = pd.read_csv('C:\\Users\\sagi\\Desktop\\Learning\\Python\\Datasets\\breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True) #modify the dataframe right away
df.drop(['id'],1, inplace=True) #dropping the id column
df
x=np.array(df.drop(['class'],1)) #features- everything except for the class
y=np.array(df['class']) #labels- just the class
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #shuffling the data and split x,y into 20% of the data

# Fitting K-NN to the Training set
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

#checking the accuracy of the model
accuracy = clf.score(x_test, y_test)
print(accuracy)


#Save the clasifier --Pickle Pickle little star
with open('K Nearest Neighbors- Sentex.pickle','wb') as f:
    pickle.dump(clf, f) #were dumping clf into f

#to use the classifier:
pickle_in=open('K Nearest Neighbors- Sentex.pickle', 'rb')
clf=pickle.load(pickle_in)

#let's predict:
example_measures=np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]]) #its all minos the id and the class
#example_measures=example_measures.reshape(2,-1) #the first '2' is the number of samples.
# -1--> In this case, the value is inferred from the length of the array and remaining dimensions
example_measures=example_measures.reshape(len(example_measures),-1) #the first '2' is the number of samples.
prediction=clf.predict(example_measures)
print(prediction)
```
