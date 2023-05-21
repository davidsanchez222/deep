import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lazypredict.Supervised import LazyRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
df = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\houseprices\train.csv", index_col=[0])

null_count = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
null_count["Index"] = [df.columns.get_loc(c) for c in null_count.index if c in df]
droplist = ["PoolQC", "MiscFeature", "Alley", "Fence"]
df = df.drop(droplist, axis=1)

cont_null_cols = ['MasVnrArea', "LotFrontage"]
disc_null_cols = ['FireplaceQu', "LotFrontage", "GarageYrBlt", 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual',
                  'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', "MasVnrType", 'Electrical']
imputer_cont = SimpleImputer(missing_values=np.nan, strategy="median")
imputer_disc = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
df.loc[:, cont_null_cols] = imputer_cont.fit_transform(df.loc[:, cont_null_cols])
df.loc[:, disc_null_cols] = imputer_disc.fit_transform(df.loc[:, disc_null_cols])

disc_cols = [c for c in df.columns if df[c].dtype in ['object', 'bool']]
disc_cols.insert(0, "MSSubClass")
disc_cols_index = [df.columns.get_loc(c) for c in disc_cols]
ct = ColumnTransformer([('non_num', OneHotEncoder(), disc_cols_index)], remainder='passthrough')
processed = ct.fit_transform(df)

feat = df.drop("SalePrice", axis=1)
label = df["SalePrice"]


v = pd.DataFrame(columns=disc_cols, data=processed)
feat_train, feat_test, label_train, label_test = train_test_split(feat, label, test_size=0.2, random_state=1)


cont_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
sc = StandardScaler()
feat_train[:, 3:] = sc.fit_transform(feat_train[:, 3:])
feat_test[:, 3:] = sc.transform(feat_test[:, 3:])
