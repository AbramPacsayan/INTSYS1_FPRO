# Libraries used
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Reading dataset
data = pd.read_csv('Salary_Data2.csv')

# First Graph: Pairplot to show the relationship between YearsExperience and Salary
sns.pairplot(data, x_vars=['YearsExperience'], y_vars=['Salary'], height=5, aspect=1.2, kind='scatter')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Relationship between Years of Experience and Salary')
plt.grid(True)
plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X = data['YearsExperience'].values.reshape(-1, 1)
y = data['Salary'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the linear regression model to the training data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict the Salary for the Test values
y_pred = lr.predict(X_test)

#  Second Graph: Actual and predicted values
plt.scatter(X_test, y_test, color='red', label='Actual Salary')
plt.plot(X_test, y_pred, color='blue', label='Predicted Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Actual vs Predicted Salary')
plt.grid(True)
plt.legend()
plt.show()

# Third Graph: Residual errors
plt.scatter(y_pred, y_test - y_pred, color='green')
plt.hlines(y=0, xmin=20000, xmax=130000, linewidth=2)
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.title('Residual Errors')
plt.grid(True)
plt.show()

# Mean Squared Error (MSE) and R-Squared (R^2) values
mse = mean_squared_error(y_test, y_pred)
rsq = r2_score(y_test, y_pred)

print('Mean Squared Error (MSE) :', mse)
print('R-Squared (R^2) :', rsq)

# Last Graph: Actual vs predicted values with a diagonal line for reference
plt.scatter(y_test, y_pred, color='red', s=20, label='Actual vs Predicted')
plt.plot(y_test, y_test, color='blue', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.grid(True)
plt.legend()
plt.show()

# Intercept and Coefficient of the line
print('Intercept of the model:', lr.intercept_)
print('Coefficient of the line:', lr.coef_)

# Accuracy Calculation
accuracy = round(rsq * 100, 2)
print("The accuracy of our model is {}%".format(accuracy))
