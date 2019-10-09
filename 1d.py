from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# Import of the data
iris = datasets.load_iris()
x = iris.data
y = iris.target

# Assignment alpha
y_pred = KMeans(n_clusters=3).fit_predict(x)

# Assignment beta
new_x = x[:,[0,1]]
y_pred = KMeans(n_clusters=3).fit_predict(new_x)

# Assignment gamma
axis = [
    [0,2],
    [0,3],
    [1,2],
    [1,3],
    [2,3]
]
for i in axis:
    new_x = x[:, i]
    y_pred = KMeans(n_clusters=3).fit_predict(new_x)
    print(y_pred)