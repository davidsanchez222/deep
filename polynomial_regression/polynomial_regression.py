import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\polynomial_regression\Position_Salaries.csv")
feat = dataset.iloc[:, 1].values
lab = dataset.iloc[:, -1].values
feat = feat.reshape(10,1)
lab = lab.reshape(10,1)
# feat_train, feat_test, lab_train, lab_test = train_test_split(feat, lab, test_size=.2, random_state=0)

lin_reg = LinearRegression()
lin_reg.fit(feat, lab)

poly_reg = PolynomialFeatures(degree=3)

feat_poly = poly_reg.fit_transform(feat)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(feat_poly, lab)

plt.scatter(feat, lab, color="red")
plt.plot(feat, lin_reg.predict(feat), color="blue")
plt.title("Truth or Bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

plt.scatter(feat, lab, color="red")
plt.plot(feat, lin_reg_2.predict(feat_poly), color="blue")
plt.title("Truth or Bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

lin_reg.predict([[6.5]])
