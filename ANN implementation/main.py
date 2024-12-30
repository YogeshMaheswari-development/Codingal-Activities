import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU, PReLU, ELU
df = pd.read_csv('https://drive.usercontent.google.com/u/0/uc?id=1wyq5A0enhFxVpc-PObzEs4aZWs4EzkjI&export=download')
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
lb = LabelEncoder()
df ['Geography'] = lb.fit_transform(df['Geography']) 
df['Gender'] = lb.fit_transform(df ['Gender'])
print(df.info())
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
print(df.shape)
y = df.pop('Exited')
x = df
print(x.shape, y.shape)
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train.shape, x_test.shape)
classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=10))
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
classifier.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])
model_hist = classifier.fit(x_train, y_train, batch_size=10, epochs=100)
classifier.summary()
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
print(y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))