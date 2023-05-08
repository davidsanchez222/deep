import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.compose import ColumnTransformer
from statistics import stdev
from keras import regularizers
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
df = pd.read_csv(r"C:\Users\David\PycharmProjects\deep\batteries\lithiumbatteries.csv")
df.drop("Materials Id", axis=1, inplace=True)

numerical_transformer = StandardScaler()
label_transformer = OrdinalEncoder()

n_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
obj_cols = [c for c in df.columns if df[c].dtype in ['object', 'bool']]
print(n_cols, obj_cols)

ct = ColumnTransformer([('num', numerical_transformer, n_cols), ('non_num', label_transformer, obj_cols)])
processed = ct.fit_transform(df)
new_df = pd.DataFrame(columns=df.columns, data=processed)

feat = new_df.drop("Crystal System", axis=1)
label = new_df["Crystal System"]

n_runs = 20
t_size = .3

score = []
for j in range(n_runs):
    feat_train, feat_test, label_train, label_test = train_test_split(np.array(feat), np.array(label), test_size=t_size, shuffle=True)
    y_encoded = to_categorical(label_train)
    model = Sequential()
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(feat_train, y_encoded, epochs=100, verbose=False, validation_split=0.2)

    pred_probs = model.predict(feat_test)
    pred_class = np.argmax(pred_probs, axis=1)
    score.append(accuracy_score(label_test, pred_class))

#%%
