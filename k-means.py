from sklearn.cluster import KMeans
import numpy as np

data = np.array([[20, 500], [40, 1000], [30, 800], [18, 300], [28, 1200], [35, 1400], [45, 1800]])
k=2

kmeans = KMeans(n_clusters = k)
kmeans.fit(data)
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_
print("Cluster Centers: ")
for i, center in enumerate(cluster_centers):
    print("Cluster ", i+1, " Center: ", center)
print("\nLabels: ")
for i,label in enumerate(labels):
    print("Data Point ", i+1, " is in Cluster ", label+1)

