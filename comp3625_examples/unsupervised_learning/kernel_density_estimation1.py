import pandas as pd
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
import numpy as np


# read in same grades
data = pd.read_csv('./comp3625_examples/unsupervised_learning/COMP 1701 grades.csv', index_col=None)

# fit a kernel density estimator
kde = KernelDensity(bandwidth=0.05)
kde.fit(X=data['Midterm 1'].to_numpy().reshape((-1, 1)))

# plot the learned distribution
grade_values = np.linspace(0, 1, 50)
log_likelihoods = kde.score_samples(grade_values.reshape((-1, 1)))

plt.plot(grade_values, np.exp(log_likelihoods))
plt.show()