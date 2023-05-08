import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\svr\Position_Salaries.csv")
feat = dataset.iloc[:, 1:-1].values
label = dataset.iloc[:, -1].values
label = label.reshape(len(label), 1)

sc_feat = StandardScaler()
sc_label = StandardScaler()
feat = sc_feat.fit_transform(feat)
label = sc_label.fit_transform(label)

regressor = SVR(kernel="rbf")
regressor.fit(feat, label)

sc_label.inverse_transform(regressor.predict(sc_feat.transform([[6.5]])).reshape(-1, 1))

plt.scatter(sc_feat.inverse_transform(feat), sc_label.inverse_transform(label), color="red")
plt.plot(sc_feat.inverse_transform(feat), sc_label.inverse_transform(regressor.predict(feat).reshape(-1, 1)), color='blue')
plt.title("Truth of Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# higher resolution and smoother curve
feat_grid = np.arange(min(sc_feat.inverse_transform(feat)), max(sc_feat.inverse_transform(feat)), 0.1)
feat_grid = feat_grid.reshape((len(feat_grid), 1))
plt.scatter(sc_feat.inverse_transform(feat), sc_label.inverse_transform(label), color="red")
plt.plot(feat_grid, sc_label.inverse_transform(regressor.predict(sc_feat.transform(feat_grid)).reshape(-1, 1)), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
