import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\advertising\advertising.csv")

dataset.drop(["Ad Topic Line", "City", "Country"], axis=1, inplace=True)

for i in range(len(dataset)):
    timestamp = dataset.iloc[i, -2]
    utc_time_index = pd.to_datetime([timestamp]).astype(int) / 10**9
    utc_time = utc_time_index[0]
    dataset.iloc[i, -2] = utc_time

feat = dataset.iloc[:, :-1]
label = dataset.iloc[:, -1]

feat_train, feat_test, label_train, label_test = train_test_split(feat, label, test_size=.2, random_state=30)

sc = StandardScaler()
feat_train = sc.fit_transform(feat_train)
feat_test = sc.fit_transform(feat_test)


logmodel = LogisticRegression()
logmodel.fit(feat_train, label_train)

predictions = logmodel.predict(feat_test)
matrix = confusion_matrix(label_test, predictions)
print(classification_report(label_test, predictions))
accuracy = accuracy_score(label_test, predictions)
#%%
