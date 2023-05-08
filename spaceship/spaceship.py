import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from IPython.display import clear_output
import lazypredict
from lazypredict.Supervised import LazyClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\spaceship\train.csv")
transported = pd.get_dummies(dataset["Transported"], drop_first=True, prefix="Transported")
homeplanet = pd.get_dummies(dataset["HomePlanet"], drop_first=True)
cryosleep = pd.get_dummies(dataset["CryoSleep"], drop_first=True, prefix="Cryosleep")
destination = pd.get_dummies(dataset["Destination"], drop_first=True)
vip = pd.get_dummies(dataset["VIP"], drop_first=True, prefix="VIP")
droplist = ["Transported", "HomePlanet", "CryoSleep", "Destination", "VIP", "Name", "Cabin", "PassengerId"]
dataset.drop(droplist, axis=1, inplace=True)
dataset = pd.concat([dataset, homeplanet, cryosleep, vip, destination, transported], axis=1)
dataset.dropna(axis=0, inplace=True)
feat = dataset.iloc[:, :-1]
label = dataset.iloc[:, -1]

feat_train, feat_test, label_train, label_test = train_test_split(feat, label, test_size=.2)

sc = StandardScaler()
feat_train = sc.fit_transform(feat_train)
feat_test = sc.transform(feat_test)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(feat_train, label_train)

regressor = LogisticRegression()
regressor.fit(feat_train, label_train)

predictions = rfc.predict(feat_test)
accuracy = accuracy_score(predictions, label_test)

validation = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\spaceship\test.csv")
validation.dropna(axis=0, inplace=False)
#%%
# 778: all rows removed with null values
#
clf = LazyClassifier(verbose=0,
                     ignore_warnings=True,
                     custom_metric=None,
                     predictions=False,
                     random_state=12,
                     classifiers='all')

models, predictions = clf.fit(feat_train, feat_test, label_train, label_test)
clear_output()

models