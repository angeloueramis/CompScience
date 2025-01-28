import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 7, 8, 10])

def linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = (sum((x - x_mean) * (y - y_mean)))/(sum((x - x_mean) ** 2))
    #y-intercept(b)
    b = y_mean - slope * x_mean

    return slope, b
slope, b = linear_regression(x, y)
#results
print(f"Slope: {slope}")
print(f"Y-intercept (b): {b}")
print(f"Equation of the line: y = {slope:.2f}x + {b:.2f}")

x_line = np.linspace(min(x), max(x), 100)
y_line = slope * x_line + b

plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x_line, y_line, color='red', label=f'Regression Line: y = {slope:.2f}x + {b:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()