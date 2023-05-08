import numpy as np
import matplotlib.pyplot
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%%
dataset = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\preprocessing\Data.csv")
X = dataset.iloc[:, :-1].values  # features
y = dataset.iloc[:, -1].values   # result; purchase? yes or no

# replacing missing values with mean of the rest of the columns values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")  # create imputer object
imputer.fit(X[:, 1:3])  # fit it to the columns with missing values
X[:, 1:3] = imputer.transform(X[:, 1:3])  # replace missing values with new transformed values

# ENCODING CATEGORICAL DATA
# encoding indepedent variable
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# encoding dependent variable
le = LabelEncoder()
y = le.fit_transform(y)

# split dataset into training and test test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


"""
Standardisation
x - mean(x)  /  stdev(x)

Normalization
x - min(x)  /  max(x) - min(x) or known as minmaxscaler

Best guessed set of training set is called loss
best weight has least loss
weight is best set of neural networks
"""

# feature scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])
#%%
