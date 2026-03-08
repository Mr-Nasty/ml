import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
iris = load_iris()
x = iris.data
df = pd.DataFrame(x, columns=iris.feature_names)
print(df.head())

# Standardize data
x_scaled = StandardScaler().fit_transform(x)

# Elbow Method
wcss = []
for k in range(1,11):
    wcss.append(KMeans( n_clusters=k,n_init=10,random_state=42).fit(x_scaled).inertia_)

plt.plot(range(1,11), wcss)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# KMeans clustering
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(x_scaled)

print(df.head())

# Plot clusters
plt.scatter(x_scaled[clusters==0,0], x_scaled[clusters==0,1])
plt.scatter(x_scaled[clusters==1,0], x_scaled[clusters==1,1])
plt.scatter(x_scaled[clusters==2,0], x_scaled[clusters==2,1])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='red')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KMeans Clustering (Iris)")
plt.show()

# Hierarchical clustering dendrogram
plt.figure()
dendrogram(linkage(x_scaled, method='ward'))
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.show()