from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Maxpooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# print(np.unique)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64,(2,2), padding='same', input_shape=(28,28,1)))
model.add(Maxpooling2D())
model.add(Conv2D(filters=64, kernel_size=(2,2),padding='valid', activation='relu'))
model.add(Conv2D(32,2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.summary()



