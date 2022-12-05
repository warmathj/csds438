# Import libraries 
from sklearn import tree
from sklearn.datasets import load_iris

# Load data 
iris = load_iris()
X = iris.data
y = iris.target

# Create decision tree classifier object 
clf = tree.DecisionTreeClassifier()

# Train model 
clf = clf.fit(X, y)

# Make predictions 
predictions = clf.predict(X)
print(predictions)