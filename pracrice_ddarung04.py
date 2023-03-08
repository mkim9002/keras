import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

#1.data
path='./keras/_data_ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print (train_csv)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)
print(test_csv.shape)

print(train_csv.columns)

print(train_csv.info())

print(train_csv.describe())


#2.모델구성
print(type(train_csv))

print(train_csv.isnull().sum)

#결축지처리
train_csv = train_csv.dropna() # 결축지 삭제
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)

###### train.csv 에서 x,y 분리##########
x = train_csv.drop(['count'], axis=1)

print(x)

y = train_csv['count']
print(y)
###### train.csv 에서 x,y 분리##########
x_train, x_test, y_train, y_test = train_test_split(
    x,y,shuffle=True, train_size=0.7, random_state=777
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#모델구성
model = Sequential()
model.add(Dense(81,input_dim=9))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련 COMPILE, fit, epochs

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs = 10, batch_size=128, verbose=1)




