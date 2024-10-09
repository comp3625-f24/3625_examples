from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np

dataset = load_iris()

# print size of X (150 cases, 4 attributes)
print(dataset.data.shape)

# print the Y vector (categories for each of the 150 flowers)
print(dataset.target)

# print the attribute names
print(dataset.feature_names)

# plot 2 of the features, along with Y
# plt.scatter(dataset.data[:,0], dataset.data[:,1], c=dataset.target)
# plt.xlabel(dataset.feature_names[0])
# plt.ylabel(dataset.feature_names[1])
# plt.show()

# try to fit a decision tree to the data
training_idx = np.random.random(150) > 0.5
tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(dataset.data[training_idx, :], dataset.target[training_idx])
# plot_tree(tree, feature_names=dataset.feature_names)
# plt.show()

# use the model to make predictions
predictions = tree.predict(dataset.data[~training_idx])
print(predictions)
print(dataset.target[~training_idx])
# print the accuracy of the tree's predictions on the test data
print((predictions == dataset.target[~training_idx]).mean())



