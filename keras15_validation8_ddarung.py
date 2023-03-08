#데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = './_data/ddarung/'   #점 하나 현재폴더의밑에 점하나는 스터디 

train_csv = pd.read_csv(path + 'train.csv' ,
                        index_col=0)
# train_csv = pd.read_csv('./_data/ddarung/train.csv')

print(train_csv)
print(train_csv.shape) # 출력결과 (1459,11)


test_csv = pd.read_csv(path + 'test.csv' ,
                       index_col=0)                 
                        
print(test_csv)
print(test_csv.shape) #1459,10
##########################################


print(train_csv.columns)
# Index(['id', 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(train_csv.info)
#  0   hour                    1328 non-null   int64
#  1   hour_bef_temperature    1328 non-null   float64
#  2   hour_bef_precipitation  1328 non-null   float64
#  3   hour_bef_windspeed      1328 non-null   float64
#  4   hour_bef_humidity       1328 non-null   float64
#  5   hour_bef_visibility     1328 non-null   float64
#  6   hour_bef_ozone          1328 non-null   float64
#  7   hour_bef_pm10           1328 non-null   float64
#  8   hour_bef_pm2.5          1328 non-null   float64
#  9   count                   1328 non-null   float64

print(type(train_csv)) #train_csv 'pandas.croe.frame.dataframe'

#########################결측치 처리#####################
#결측치 처리 1 .제거
# pirnt(train_csv.insul11())
print(train_csv.isnull().sum())
train_csv = train_csv.dropna() ####결측치 제거#####
print(train_csv.isnull().sum()) #(1328, 10)
print(train_csv.info())
print(train_csv.shape)



############################## train_csv 데이터에서 x와y를 분리
x = train_csv.drop(['count','casual','registered'], axis=1) #2개 이상 리스트 
print(x)

y = train_csv['count']
print(y)
############################## train_csv 데이터에서  x와y를 분리
x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, train_size=0.7, random_state=777
)

print(x_train.shape, x_test.shape) #(1021,9)(438,9) -> (929.9) (399 9)
print(y_train.shape, y_test.shape) #(1021,)(438,) (929) (399,)


# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=8))




#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32, 
          verbose=1,validation_split=0.2)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

#17의 예측값:  [[16.999998]]