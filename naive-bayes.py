import numpy as np

#X - samples Y-features
X = np.array([[-1,-1], [-2, -1], [-3, -2], [1,1], [3,2], [2, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
#import model
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
#X - samples Y-features
clf.fit(X, Y)
#predict values
print(clf.predict([[-3, -1]]))
print(clf.predict([[3, 1]]))
