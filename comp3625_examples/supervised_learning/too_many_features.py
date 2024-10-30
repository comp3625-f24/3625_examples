import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# a completely random training set
x = np.random.random(size=(10,100))
y = np.random.random(size=(10,1))

# fit a linear regression model to the training set
model = LinearRegression()
model.fit(x, y)

# print out the model's coefficients
print(sorted(np.abs(model.coef_)[0]))

# get the model's predictions on the training set
predictions = model.predict(x)
print('r2=', r2_score(predictions, y))