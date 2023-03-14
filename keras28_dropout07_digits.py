#사이킷런 load_digits

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.metrics import accuracy_score #분류=> 결과지표 'accuracy_score' 떠올라야함
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터 
datasets = load_digits()
# print(datasets.DESCR) #(1797, 64)  #pandas : describe()
# print(datasets.feature_names)  #pandas : colums()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(1797, 64) (1797,)
print(x)
print(y)  
print('y의 라벨값 :', np.unique(y))  #y의 라벨값 : [0 1 2 3 4 5 6 7 8 9]
print(np.unique(y, return_counts=True))  


scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)



##########데이터 분리전에 one-hot encoding하기##########################
#y값 (1797,) ->  (1797,10) 만들어주기
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)
print(y.shape) #(1797, 10)
##################################################################


#데이터분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, 
    train_size=0.8,
    stratify=y    
)
print(y_train)                                  
print(np.unique(y_train, return_counts=True))  

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=64))
model.add(Dropout(0.3))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax')) 
# #3개의 데이터를 뽑으니까 *label의 개수만큼 노드를 잡는다!! 

# input1 = Input(shape=(64,))
# dense1 = Dense(8, activation = 'relu')(input1)
# dense2 = Dense(4, activation = 'relu')(dense1)
# dense3 = Dense(8, activation = 'relu')(dense2)
# dense4 = Dense(4, activation = 'relu')(dense3)
# output1 = Dense(10, activation = 'softmax')(dense4)
# model = Model(inputs=input1, outputs=output1)




#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

#EarlyStopping추가
es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=16,
          validation_split=0.2,
          verbose=1,
          )


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results:', results)  
y_predict = np.round((model.predict(x_test)))
print(y_predict)

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)