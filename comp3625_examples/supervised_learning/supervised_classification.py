from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
pd.options.display.width = 0

# ______________________________________________________________________________________________________________
# In this lab we will play with some supervised learning algorithms in the SKLearn package
# ______________________________________________________________________________________________________________


# load the "wine" dataset into a dataframe
# It contains 178 instances of wines, with various features. Each is classified into 1 of 3 categories
df = pd.read_csv('wine.csv', index_col=None)

# separate into features (X) and classes (Y)
x = df.drop('class', axis=1)
y = df['class']

# print out a statistical summary of the feature values
print(x.describe())

# instantiate a k-nearest-neighbor classifier
model = KNeighborsClassifier(n_neighbors=5)

# now fit the KNN classifier to the data using the "fit" method
# you may want to refer to the documentation here: https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
model.fit(x, y)


# now that the classifier has been built, test it out on the training data
# use the "predict" method to generate predictions for the training data in X
# measure the prediction's % accuracy (# predictions right / total predictions)
predictions = model.predict(x)
accuracy = (predictions == y).sum() / len(predictions)
print(f'accuracy={accuracy}')


# the K nearest neighbors are identified based on euclidean distance between feature values
# are all the features treated equally in this distance calculation? Refer back to the summary stats - could any of
# the features be skewing the distance calcs?
# if so, try to correct this by standardizing each column of X (divide each column by its standard deviation)
# then re-fit and re-predict
x = x / x.std()


# now suppose we want to see how confident the model is in its predictions
# use the "predict_proba" method instead of "predict". This method returns a probability distribution over the 3 classes
# instead of a hard classification. Extract the max probability for each instance to see how confident the model is
# YOUR CODE HERE


# Measuring models' prediction accuracy on the training data will always be optimistic
# Split the dataset into "train" and "test" subsets, re-fit the model to the "train" subset, and test its accuracy
# on the "test" subset. This gives a better estimate of how the model will perform on new cases
# YOUR CODE HERE


# Now try using a LogisticRegression instead of the KNN classifier. How does the simple linear model do on this data
# compared to the KNN approach?
# YOUR CODE HERE


# Now do these follow-up exerices, in any order:
#   -   cross validation is better than a simple train/test split. Implement a cross validation to measure your model's
#       performance
#
#   -   KNN is just one of many supervised classification algorithms. See the full list of classifiers in SKLearn:
#       https://scikit-learn.org/1.5/supervised_learning.html
#       and check out https://scikit-learn.org/1.5/auto_examples/classification/plot_classifier_comparison.html
#       Try out the LogisticRegression, SVM, Decision Tree, and Random Forest. Read a bit about each one in the user
#       guide (https://scikit-learn.org/1.5/supervised_learning.html). Which one performs best on this data?
#
#   -   The "diabetes" dataset contains baseline clinical measurements from 442 diabetes patients. The target value
#       shows progression of the disease 1 year after baseline. The target is numeric, so this is a regression problem
#       instead of classification. Copy a similar workflow as above, but try SKLearn's KNeighborsRegressor and
#       LinearRegression models on the diabetes dataset.
