import pandas as pd
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

#read and extract data
dataset = pd.read_csv('CC.csv')
x = dataset.iloc[:,[1,2,3,4]]

#find means of rows
x.apply(lambda x: x.fillna(x.mean()), axis=0)

#kmeans clustering loop
wcss = []
for i in range(1,11):
     kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
     kmeans.fit(x)
     wcss.append(kmeans.inertia_)

#plot data
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

#elbow method determines number of clusters
nclusters = 4 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

y_cluster_kmeans = km.predict(x)

#silhouette score evaluates distance of cluster
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)

#dimension reduction using principal component analysis (performs linear mapping of data, decreases variance)
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)


x_scaler = scaler.transform(x)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)

#repeat kmeans clustering several times to obtain best results
nclusters = 4 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

y_cluster_kmeans = km.predict(x)

# evaluate silhouette score (ideally closer to +1)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)
