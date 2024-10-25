import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")


# ______________________________________________________________________________________________________________
# In this exercise we will write a hill-climbing search to find the best subset of features to use for learning
# ______________________________________________________________________________________________________________


# load the "wine" dataset into a dataframe
# It contains 178 instances of wines, with various features. Each is classified into 1 of 3 categories
df = pd.read_csv('wine.csv', index_col=None)

# randomly assign instances to training and test sets (50% each for illustration purposes - 66.6% training maybe more typical)
train_idx = np.random.random(size=df.shape[0]) < 0.5

# separate into features (X) and classes (Y), for training and test sets
x_train = df.drop('class', axis=1).loc[train_idx]
y_train = df['class'].loc[train_idx]
x_test = df.drop('class', axis=1).loc[~train_idx]
y_test = df['class'].loc[~train_idx]

# standardize the feature values (to solve the issues with distance measurement identified in the last lab!)
x_train /= x_train.std()
x_test /= x_test.std()


# The dataset includes 13 features. Some may be redundant, or irrelevant to the prediction of wine category.
# Let's find the best subset of features to use for supervised learning
# write a function that accepts a list of feature names to use, instantiates a K-Nearest_Neighbors classifier,
# fits it to that subset of the data, and then measures its prediction accuracy on the test set.
# Note that you can get the complete list of feature names like this: x_train.columns, or as a set: set(x_train.columns)
# and you can reduce the dataframe to a subset of features like this: x_train_subset = x_train[desired_features_list]

def model_accuracy(features) -> float:
    # YOUR CODE HERE


# call your function with all features included. What is the accuracy?
# YOUR CODE HERE


# Now write a hill-climbing search that searches through feature subsets, finding the subset that gives the best performance
# You can search in either direction:
#   Forward: Start with an empty set of features, and add features one-by-one, always adding the feature that gives the
#               best improvement in accuracy
#   Backward: Start the set of all features, and remove one-by-one.
# In either case, stop the search when the next addition/deletion doesn't improve the accuracy
# YOUR CODE HERE


# What is the best subset accuracy? How big is the subset compared to the original 13 features?

# Now try the whole thing with a LogisticRegression instead of K-nearest-neighbors (should be a 1-line chance, just instantiate
# a LogisticRegression() instead of a KNearestNeighborsClassifier). Are the results different?
# (remember, fewer features often means less chance of overfitting, so a slight drop in accuracy for a large reduction
# in # of features is probably a good trade-off)