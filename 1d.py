from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Import of the data
iris = datasets.load_iris()
x = iris.data
y = iris.target

def good_pred(prediction, target=y):
    permutations = list(itertools.permutations([0, 1, 2]))
    result=0
    for i in permutations:
        good_predictions=0
        for j in range(len(prediction)):
            if(prediction[j] == 0 and target[j]== i[0]) :
                good_predictions=good_predictions+1
            if(prediction[j] == 1 and target[j] == i[1]):
                good_predictions = good_predictions + 1
            if(prediction[j] == 2 and target[j] == i[2]):
                good_predictions = good_predictions + 1
        result = max(result , good_predictions)
    return result


# Assignment alpha
y_pred = KMeans(n_clusters=3).fit_predict(x)
print('Alpha: Wrong labeled points in clusters: ',150-good_pred(y_pred))



# Assignment beta
new_x = x[:, [0, 1]]
y_pred_new = KMeans(n_clusters=3).fit_predict(new_x)
print('Beta: Wrong labeled points in clusters: ',150-good_pred(y_pred_new))


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
    y_pred_new = KMeans(n_clusters=3).fit_predict(new_x)
    print('Gamma: Wrong labeled points in clusters (axis-parallel onto ', i , '): ', 150 - good_pred(y_pred_new))



