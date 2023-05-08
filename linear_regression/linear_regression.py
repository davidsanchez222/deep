import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\linear_regression\Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# y_test is real salaries, y_pred is predicted salaries
y_pred = regressor.predict(X_test)

# visualizing training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs. Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# visualizing test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")  # <- training and test set use same regression line
plt.title("Salary vs. Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


df2 = pd.DataFrame([y_pred, y_test]).T
df2.columns = ["Predictions", "Actuals"]
df2.plot(x="Predictions", y="Actuals", kind="scatter", figsize=(13,7))
plt.show()
df2["Difference"] = y_pred - y_test
df2["Difference_sq"] = df2["Difference"]**2
loss = np.sqrt(df2["Difference_sq"].mean())

