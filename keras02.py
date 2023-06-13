# 1. 데이타
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])
###
#. 모델 구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1,input_dim=1))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100)

# loss : 0.6966
