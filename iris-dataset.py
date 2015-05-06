from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

iris = load_iris()
type(iris)
print iris.data
print iris.feature_names
print iris.target
print iris.target_names

print type(iris.data)
print type(iris.target)

print iris.data.shape
print iris.target.shape

X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=1)
print knn
knn.fit(X, y)
knn.predict([3, 5, 4, 2])
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)


# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
logreg.predict(X_new)
