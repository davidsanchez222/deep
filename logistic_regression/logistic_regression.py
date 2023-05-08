import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\logistic_regression\Social_Network_Ads.csv")
feat = dataset.iloc[:, :-1].values
label = dataset.iloc[:, -1].values

feat_train, feat_test, label_train, label_test = train_test_split(feat, label, test_size=.25, random_state=0)

sc = StandardScaler()
feat_train = sc.fit_transform(feat_train)
feat_test = sc.fit_transform(feat_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(feat_train, label_train)

classifier.predict(sc.transform([[30, 87000]]))

label_pred = classifier.predict(feat_test)
np.set_printoptions(precision=2)
print(np.concatenate((label_pred.reshape(len(label_pred), 1), label_test.reshape(len(label_test), 1)), 1))

confusion_matrix(label_test, label_pred)
accuracy_score(label_test, label_pred)
