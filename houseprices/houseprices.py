import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from IPython.display import clear_output
import lazypredict
from lazypredict.Supervised import LazyRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
df = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\houseprices\train.csv")

null_count = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
null_count["Index"] = [df.columns.get_loc(c) for c in null_count.index if c in df]
droplist = ["Id", "PoolQC", "MiscFeature", "Alley", "Fence"]
df.drop(droplist, axis=1, inplace=True)

# imputing missing values use simple impute save a lot of lines
df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0], inplace=True)
df["LotFrontage"].fillna(df["LotFrontage"].median(), inplace=True)
df["GarageYrBlt"].fillna(df["YearBuilt"], inplace=True)
df['GarageCond'].fillna(df['GarageCond'].mode()[0], inplace=True)
df['GarageType'].fillna(df['GarageType'].mode()[0], inplace=True)
df['GarageFinish'].fillna(df['GarageFinish'].mode()[0], inplace=True)
df['GarageQual'].fillna(df['GarageQual'].mode()[0], inplace=True)
df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace=True)
df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0], inplace=True)
df['BsmtQual'].fillna(df['BsmtQual'].mode()[0], inplace=True)
df['BsmtCond'].fillna(df['BsmtCond'].mode()[0], inplace=True)
df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0], inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)
df["MasVnrType"].fillna(df["MasVnrType"].mode()[0], inplace=True)
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)

# taking care of discrete variables that are ints
pd.get_dummies(df["MSSubClass"], drop_first=True)

encoder = OneHotEncoder(handle_unknown="ignore")
