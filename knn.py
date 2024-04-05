from sklearn.neighbors import KNeighborsClassifier
import numpy as np

x = np.array([[2, 3], [3, 4], [5, 6], [7, 8], [10, 10]])
y = np.array(['A', 'A', 'B', 'B', 'A'])

knn_regressors = KNeighborsClassifier(n_neighbors = 3)
knn_regressors.fit(x, y)
y_pred = knn_regressors.predict([[6, 7]])
print("Prediction: ", y_pred)