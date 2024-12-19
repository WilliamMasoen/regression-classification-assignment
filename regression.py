import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

# PART 1: Basic Linear Regression
# Predict restaurant profit based on the population of the city.

# Load data from CSV file into a pandas DataFrame and assign column names 'X' (population) and 'y' (profit)
file_path_part1 = "RegressionData.csv"
data_part1 = pd.read_csv(file_path_part1, header=None, names=['X', 'y'])

# Convert data into numpy arrays and reshape for compatibility with sklearn
X_part1 = np.array(data_part1["X"]).reshape(-1, 1)
y_part1 = np.array(data_part1["y"]).reshape(-1, 1)

# Plot the data to visualize the relationship between population and profit
plt.scatter(X_part1, y_part1, color='blue', label="Actual Data")
plt.xlabel("Population (in thousands)")
plt.ylabel("Profit (in thousands)")
plt.title("Population vs. Profit")
plt.legend()
plt.show()

# Perform linear regression
reg = linear_model.LinearRegression()
reg.fit(X_part1, y_part1)

# Generate predictions and plot the linear regression fit
y_pred_part1 = reg.predict(X_part1)
plt.scatter(X_part1, y_part1, color='blue', label="Actual Data")
plt.plot(X_part1, y_pred_part1, color='red', label="Regression Line")
plt.xlabel("Population (in thousands)")
plt.ylabel("Profit (in thousands)")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

# Print the regression equation and prediction for a city with 18,000 inhabitants
print(f"The linear relationship is modeled as: y = {reg.intercept_[0]:.2f} + {reg.coef_[0][0]:.2f} * X")
print(f"The predicted profit for a city with 18,000 inhabitants is: {reg.predict(np.array([[18]]))[0][0]:.2f}")

# PART 2: Logistic Regression
# Predict hiring decisions based on exam scores.

# Load data and assign column names
file_path_part2 = "LogisticRegressionData.csv"
data_part2 = pd.read_csv(file_path_part2, header=None, names=['Score1', 'Score2', 'y'])

# Separate features (Score1, Score2) and labels (y)
X_part2 = np.array(data_part2[["Score1", "Score2"]])
y_part2 = np.array(data_part2["y"])

# Visualize the data with different markers and colors based on class labels
markers = ['o', 'x']
colors = ['hotpink', '#88c999']
plt.figure()
for i in range(len(data_part2)):
    plt.scatter(data_part2['Score1'][i], data_part2['Score2'][i], marker=markers[data_part2['y'][i]], color=colors[data_part2['y'][i]])
plt.xlabel("Score 1")
plt.ylabel("Score 2")
plt.title("Applicant Exam Scores and Hiring Decisions")
plt.show()

# Train logistic regression classifier
log_reg = linear_model.LogisticRegression()
log_reg.fit(X_part2, y_part2)

# Predict class labels and visualize classification performance
y_pred_part2 = log_reg.predict(X_part2)
colors_pred = ['red', 'blue']
plt.figure()
for i in range(len(data_part2)):
    plt.scatter(data_part2['Score1'][i], data_part2['Score2'][i], marker=markers[y_pred_part2[i]], color=colors_pred[y_pred_part2[i]])
plt.xlabel("Score 1")
plt.ylabel("Score 2")
plt.title("Predicted Hiring Decisions")
plt.show()

# PART 3: Multi-Class Classification
# Explanation of One-vs-Rest and One-vs-One methods for multi-class classification

"""
One-vs-Rest (OvR):
Each class is trained against all other classes as a binary classification problem. For predictions, 
each binary classifier outputs a score, and the class with the highest score is chosen as the prediction.

One-vs-One (OvO):
Pairs of classes are trained against each other in binary classification problems. Each classifier votes 
for one of the two classes it was trained on. The class with the most votes across all classifiers is selected.
"""
