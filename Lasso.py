# min_{w} ||Xw-y||^2_2

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

np.random.seed(42)
n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0
y = np.dot(X, coef)
y += 0.01 * np.random.normal(size=n_samples)

from sklearn import datasets, linear_model
