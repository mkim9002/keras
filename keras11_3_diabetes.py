from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,
        train_size=0.7, shuffle = True, random_state=123456789)

print(x.shape, y.shape) #(442, 10) (442,)

#2. 모델구성
model = Sequential()
model.add(Dense(19, input_dim=10))
model.add(Dense(987))
model.add(Dense(999))
model.add(Dense(2))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(1))

#컴파일 , 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs = 50, batch_size=15)

#4.평가,에측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test)          #이렇게 하면 새로운 X 값을 넣어 y값을 예측할 수 있게 된다

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

# 실습
# R2 0.62 이상
#실험결과 r2 score : 0.5319684418497093

