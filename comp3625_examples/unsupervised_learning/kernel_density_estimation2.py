from sklearn.neighbors import KernelDensity
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import random

digits = load_digits()

fig, ax = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        ax[i][j].imshow(digits.data[random.randint(0, len(digits.data))].reshape((8,8)), cmap='binary')
