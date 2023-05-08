import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\multiplelinear_regression\50_Startups.csv")
feat = dataset.iloc[:, :4].values
lab = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
feat = np.array(ct.fit_transform(feat))

feat_train, feat_test, lab_train, lab_test = train_test_split(feat, lab, test_size=.2, random_state=0)

regressor = LinearRegression()
regressor.fit(feat_train, lab_train)

lab_pred = regressor.predict(feat_test)
np.set_printoptions(precision=2)
np.concatenate((lab_pred.reshape(len(lab_pred), 1), lab_test.reshape(len(lab_test), 1)), 1)
