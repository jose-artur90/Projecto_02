import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import dataAnalysis as da

x = da.final_Hotel['price_for_resort_hotel']
y = da.final_Hotel['price_for_city_hotel']

fig = plt.figure(figsize=(25,20))
plt.scatter(x, y)
fig.savefig("cluster01.png")
plt.show()
#Two  lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()   

###

data = list(zip(x, y))
print(data)

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

fig = plt.figure(figsize=(25,20))
plt.scatter(x, y, c=kmeans.labels_)
fig.savefig("cluster02.png")
plt.show()

#Two  lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()


##
inertias = []

for i in range(1,13):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

fig = plt.figure(figsize=(25,20))
plt.plot(range(1,13), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
fig.savefig("cluster03.png")
plt.show()

#K=3
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

fig = plt.figure(figsize=(25,20))
plt.scatter(x, y, c=kmeans.labels_)
fig.savefig("cluster04.png")
plt.show()
