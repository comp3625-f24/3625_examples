import pandas as pd
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
import numpy as np


data = pd.read_csv('COMP 1701 grades.csv', index_col=None)

kde = KernelDensity(bandwidth=0.05)
kde.fit(X=data['Midterm 1'].to_numpy().reshape((-1, 1)))