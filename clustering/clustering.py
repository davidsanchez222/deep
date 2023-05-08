import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\clustering\Mall_Customers.csv")
feat = dataset.iloc[:, [3, 4]].values


wcss = []
for x in range(1, 11):
    kmeans = KMeans(n_clusters=x, init="k-means++", random_state=42)
    kmeans.fit(feat)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()