# K-Means Clustering with Scikit-Learn
```python
'''
"In K means we forms clusters of data based on the similarity between data instances.
For this particular algorithm to work, the number of clusters has to be defined beforehand.
The K in the K-means refers to the number of clusters.
The K-means algorithm starts by randomly choosing a centroid value for each cluster.
After that the algorithm iteratively performs three steps:
(i) Find the Euclidean distance between each data instance and centroids of all the clusters; (
ii) Assign the data instances to the cluster of the centroid with nearest distance;
(iii) Calculate new centroid values based on the mean values of the coordinates of all the data instances from the corresponding cluster."
'''

#Import Libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Preparing the data

#We'll use 10 rows and 2 columns.
'''We create a numpy array of data points because the Scikit-Learn library can work with numpy array type data inputs without requiring any preprocessing'''

x=np.array([[5,3],
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91],])

#Visualize the Data
#plotting all the values in the first column of the X array against all the values in the second column

plt.scatter(x[:,0],x[:,1], label='True Position')

#Creating Clusters

kmeans=KMeans(n_clusters=3 )#create a KMeans object
kmeans.fit(x) #call the fit method on kmeans and pass the data that we want to cluster
print(kmeans.cluster_centers_)#printing the centroids--> first row is the first centroid.
print(kmeans.labels_) #which point correspond to which label (centroid)

plt.scatter(x[:,0],x[:,1], c=kmeans.labels_, cmap='rainbow') #plotting the new scatter with the labels
#the kmeans.labels_ passes value for the c parameter that corresponds to labels

plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black') #plotting the centroid for each cluster
```



