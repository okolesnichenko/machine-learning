import numpy as np

#X - samples Y-features
X = np.array([[-1,-1], [-2, -1], [-3, -2], [1,1], [3,2], [2, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
#import model
from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
clf.fit(X, Y)
print(clf.predict([[-0.8,-1]]))
