import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
train = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\titanic\train.csv")

# visualizing null values
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# boxplot of Pclass and age
plt.figure(figsize=(10, 7))
sns.boxplot(x="Pclass", y="Age", data=train)
plt.show()

# find average age by class
train.groupby(["Pclass"]).mean()["Age"]
# Output:
# Pclass
# 1    38.233441
# 2    29.877630
# 3    25.140620
# Name: Age, dtype: float64

# impute null values with average age of Pclass
for i in range(len(train["Age"])):
    if pd.isnull(train.iloc[i, 5]):
        if train.iloc[i, 2] == 1:
            train.iloc[i, 5] = 38.23
        if train.iloc[i, 2] == 2:
            train.iloc[i, 5] = 29.88
        else:
            train.iloc[i, 5] = 25.14

# drop cabin column; majority of data is null
# embarked has one null so lets drop that too
train.drop("Cabin", axis=1, inplace=True)
train.dropna(inplace=True)

sex = pd.get_dummies(train["Sex"], drop_first=True)
embark = pd.get_dummies(train["Embarked"], drop_first=True)
pclass = pd.get_dummies(train["Pclass"], drop_first=True)
pclass.columns = ["class2", "class3"]
train = pd.concat([train, sex, embark, pclass], axis=1)
train.drop(["PassengerId", "Name", "Sex", "Embarked", "Ticket"], axis=1, inplace=True)

feat = train.drop(["Survived"], axis=1)
label = train.iloc[:, 0]
feat_train, feat_test, label_train, label_test = train_test_split(feat, label, test_size=.3, random_state=42)

sc = StandardScaler()
feat_train = sc.fit_transform(feat_train)
feat_test = sc.fit_transform(feat_test)

logmodel = LogisticRegression()
logmodel.fit(feat_train, label_train)

predictions = logmodel.predict(feat_test)

print(classification_report(label_test, predictions))
matrix = confusion_matrix(label_test, predictions)
accuracy = accuracy_score(label_test, predictions)
#%%
# .798: base model
# .801: replaced categorical pclass variable with dummy variable:
# .816: added StandardScaler.fit_transform to features
# .798: class_weight="balanced" to regressor
