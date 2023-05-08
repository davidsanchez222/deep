import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\knn\KNN_Project_Data")

sc = StandardScaler()
scaled_features = sc.fit_transform(dataset.iloc[:, :-1])
label = dataset.iloc[:, -1]

feat_train, feat_test, label_train, label_test = train_test_split(scaled_features, label, train_size=.2, random_state=30)

error_rate = []
for i in range(1, 60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(feat_train, label_train)
    predictions = knn.predict(feat_test)
    error_rate.append(np.mean(predictions != label_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 60), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.show()

knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(feat_train, label_train)
predictions = knn.predict(feat_test)

print(classification_report(label_test, predictions))
print(confusion_matrix(label_test, predictions))
accuracy = accuracy_score(label_test, predictions)
