import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, 4, 6, 8])
y = np.array([3, 7, 5, 10])

slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept
print("Slope: ", slope)
print("Intercept: ", intercept)

plt.scatter(x, y, label = "Original Data")
plt.plot(x, regression_line, color="red", label = "Regression Line")

plt.legend()
plt.show()