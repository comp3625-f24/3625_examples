import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import minimize, dual_annealing


# ______________________________________________________________________________________________________________
# In this exercise we will use optimization functions to optimize a decision tree's meta-parameters
# ______________________________________________________________________________________________________________


# load the "wine" dataset into a dataframe
# It contains 178 instances of wines, with various features. Each is classified into 1 of 3 categories
df = pd.read_csv('./comp3625_examples/supervised_learning/wine.csv', index_col=None)

# randomly assign instances to training and test sets (50% each for illustration purposes - 66.6% training maybe more typical)
train_idx = np.random.random(size=df.shape[0]) < 0.5

# separate into features (X) and classes (Y), for training and test sets
x_train = df.drop('class', axis=1).loc[train_idx]
y_train = df['class'].loc[train_idx]
x_test = df.drop('class', axis=1).loc[~train_idx]
y_test = df['class'].loc[~train_idx]

# We will manipulate the following parameters of the SKlearn decision tree algorithm:
#   min_impurity_decrease: threshold between 0-1 on how much improvement a split must offer. Defaults to 0 in SKLearn's decision tree.
#   ccp_alpha: parameter (between 0-1) that controls amount of pruning (removal of less-useful subtrees). Defaults to 0 in SKLearn's decision tree.
# write a function that accepts a tuple containing (min_impurity_decrease, ccp_alpha) values, instantiates a decision
# tree with those params, fits it to that subset of the data, and then measures its prediction accuracy on the test set.
# refer to docs if needed: https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# optimization algorithms usually minimize things, so return the negative of the accuracy

def tree_accuracy(params) -> float:
    min_impurity_decrease, ccp_alpha = params
    tree = DecisionTreeClassifier(
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha
        )
    tree.fit(x_train, y_train)
    predictions = tree.predict(x_test)
    accuracy = (predictions == y_test).mean()
    return -accuracy

# call your function with the params (0, 0) - these are the decision tree defaults. What is the default accuracy?
print(tree_accuracy((0.5, 0.5)))


# now use scipy's local-search and optimization functions to find the tree params giving the best performance.
# specifically use,
# dual_annealing: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html
# and also try
# Nelder-Mead, via the "minimize" function: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
# Both these functions have a lot of parameters, but you'll only need to specify func, x0, and bounds
# (and method="Nelder-Mead" for the Nelder Mead approach)
results1 = dual_annealing(tree_accuracy, x0=(0.5,0.5), bounds=[(0, 1), (0, 1)])
print(results1)
results2 = minimize(tree_accuracy, x0=(0.5,0.5), bounds=[(0, 1), (0, 1)], method="Nelder-Mead")
print(results2)

# print the results returned by these functions. Did the search improve the accuracy? What were the best
# min_impurity_decrease and ccp_alpha values found?
# There's some randomness in the training/test split, so you can run several times and see if the results change.
# How many times did each optimizer call your function (i.e. how many trees were built?). Trees build fast on small datasets,
# so making many function calls was likely not a problem here. But for a more complex supervised learning algorithm, it may
# become a problem...
