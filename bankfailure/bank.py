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
df = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\bankfailure\bankdata.csv", parse_dates=['FAILDATE'])

df.shape[0]
# 4104

df["STATE"] = df.apply(lambda row: row["CITYST"][-2:], axis=1)
df["STATE"].value_counts()[:3]
# TX    910
# CA    263
# IL    227

