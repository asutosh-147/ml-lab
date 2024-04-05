import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X_pos = np.array([[2, 2], [2, -2], [-2, -2], [-2, 2]])
y_pos = np.ones(len(X_pos))
X_neg = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
y_neg = -np.ones(len(X_neg))
X = np.concatenate((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))

# Modify points based on the condition
for i, (x1, x2) in enumerate(X):
    if x1 ** 2 + x2 ** 2 > 4:
        X[i][0] = 4 - x2 + abs(x1 - x2)
        X[i][1] = 4 - x1 + abs(x1 - x2)

svm = SVC(kernel='linear')
svm.fit(X, y)
support_vectors = svm.support_vectors_
w_t = (svm.coef_[0])
w=[round(coeff) for coeff in w_t]
bias = round(svm.intercept_[0])

print("Weight:",w)
print("Bias:",bias)

plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Positive')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Negative')

# Plot decision boundary
if w[1] != 0:
    x_values = np.linspace(-4, 4, 10)
    y_values = (-bias - w[0] * x_values) / w[1]
    plt.plot(x_values, y_values, color='black', linestyle='-', label='Decision Boundary')
else:
    plt.axvline(x=-bias / w[0], color='black', linestyle='-', label='Decision Boundary')

plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='green', marker='s', label='Support Vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Linear SVM Classifier')
plt.legend()
plt.grid(True)
plt.show()