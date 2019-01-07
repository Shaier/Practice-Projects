# Hierarchical Clustering with Python and Scikit-Learn
```python
"Hierarchical clustering is a type of unsupervised machine learning algorithm used to cluster unlabeled data points.
Like K-means clustering, hierarchical clustering also groups together the data points with similar characteristics
2 types of hierarchical clustering: Agglomerative (bottom-up approach starting with individual data points)
and Divisive (top-down approach-all the data points are treated as one big cluster and then you divide it into smaller ones)

Steps to Perform Hierarchical Clustering
Agglomerative clustering:
1.At the start, treat each data point as one cluster. Therefore, the number of clusters at the start will be K, while
K is an integer representing the number of data points.
2.Form a cluster by joining the two closest data points resulting in K-1 clusters.
3.Form more clusters by joining the two closest clusters resulting in K-2 clusters.
4.Repeat the above three steps until one big cluster is formed.
5.Once single cluster is formed, dendrograms are used to divide into multiple clusters depending upon the problem.

Options to measure distance between two clusters:
Measure the distance between the closes points of two clusters.
Measure the distance between the farthest points of two clusters.
Measure the distance between the centroids of two clusters.
Measure the distance between all possible combination of points between the two clusters and take the mean."


#Example 1

#Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Dataset
X = np.array([[5,3],
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])


#Plotting the data
import matplotlib.pyplot as plt

labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

#Plotting dendogram
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(X, 'single')

labelList = range(1, 11)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
#affinitiy is the distance between the datapoints, "ward" minimizes the variant between the clusters.

print(cluster.labels_) #gives us the cluster classification per point

#Plot clusters
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
#######################################################################

#Example 2


path='C:\\Users\\sagi\\Desktop\\Learning\\ML\\Datasets\\shopping_data.csv'
customer_data = pd.read_csv(path)

#Explore the data
customer_data.shape
customer_data.head()
customer_data.describe()

#Preprocessing the data
data = customer_data.iloc[:, 3:5].values
data


#Creating a dendogram to see how many clusters we have
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
"dendrogram method which takes the value returned by the linkage method of the same class."
"The linkage method takes the dataset and the method to minimize distances as parameters."
"we used ward' as the method since it minimizes then variants of distances between the clusters"

#Group points to clusters
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

#plotting the clusters
plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')

```
# Resources
https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
