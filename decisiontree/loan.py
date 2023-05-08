import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\decisiontree\loan_data.csv")
dataset.isnull().values.any()
dataset = pd.concat([dataset, pd.get_dummies(dataset["purpose"], drop_first=True)], axis=1)
dataset.drop(["purpose"], axis=1, inplace=True)

sc = StandardScaler()
scaled_features = sc.fit_transform(dataset.iloc[:, 1:])
# feat = dataset.iloc[:, 1:]
label = dataset.iloc[:, 0]

feat_train, feat_test, label_train, label_test = train_test_split(scaled_features, label, test_size=.3)
# feat_train, feat_test, label_train, label_test = train_test_split(feat, label, test_size=.3)

# sc = StandardScaler()
# feat_train = sc.fit_transform(feat_train)
# feat_test = sc.transform(feat_test)

logmodel = LogisticRegression()
logmodel.fit(feat_train, label_train)
predictions = logmodel.predict(feat_test)
accuracy = accuracy_score(predictions, label_test)
# scale after, fit_transform train and test: .903
# scale after, fit_transform train and transform test: .894
# scale beforehand: .908
#%%

dtree = DecisionTreeClassifier()
dtree.fit(feat_train, label_train)

predictions = dtree.predict(feat_test)
accuracy = accuracy_score(predictions, label_test)
#%%
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(feat_train, label_train)

rfc_pred = rfc.predict(feat_test)
rfc_accuracy = accuracy_score(rfc_pred, label_test)


