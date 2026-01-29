from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib import style

# Simple Linear Regression from scratch
xs = np.array([1, 2, 3, 4, 5, 9,12, 20], dtype=np.float64)

ys = np.array([5, 4, 6, 5, 6, 7, 15, 18], dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
          ((mean(xs) ** 2) - mean(xs ** 2)) ) # slope
    b = mean(ys) - m * mean(xs) # intercept
    return m, b

m = best_fit_slope_and_intercept(xs, ys)[0] # slope
b = best_fit_slope_and_intercept(xs, ys)[1] # intercept
regression_line = [(m*x) + b for x in xs] # y = mx + b

print("slope (m):", m)
print("intercept (b):", b)

# getting the q,ount of y distance from the regression line
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

 #predicting a value of X 6
predict_x = 6
# calculating corresponding value of Y
predict_y = (m * predict_x) + b
print(f"Predicted y for x={predict_x}: {predict_y}")

r_squqred = coefficient_of_determination(ys, regression_line)
print(f"Coefficient of Determination (R^2): {r_squqred}")

# plotting the regression line
style.use('ggplot')
plt.scatter(xs, ys, color='blue', label='Data Points')
# Plotting the prediction point
plt.scatter(predict_x, predict_y, color='green', label='Prediction')
plt.plot(xs, regression_line, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

