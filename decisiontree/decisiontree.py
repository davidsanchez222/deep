import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\decisiontree\kyphosis.csv")
dataset["Kyphosis"] = pd.get_dummies(dataset["Kyphosis"], drop_first=True)

sc = StandardScaler()
scaled_features = sc.fit_transform(dataset.iloc[:, 1:])
label = dataset.iloc[:, 0]

sns.pairplot(dataset, hue="Kyphosis")
plt.show()

feat_train, feat_test, label_train, label_test = train_test_split(scaled_features, label, test_size=.3)

dtree = DecisionTreeClassifier()
dtree.fit(feat_train, label_train)

predictions = dtree.predict(feat_test)
accuracy = accuracy_score(predictions, label_test)

rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(feat_train, label_train)

rfc_pred = rfc.predict(feat_test)
accuracy = accuracy_score(rfc_pred, label_test)