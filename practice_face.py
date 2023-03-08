import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

##########데이터 로드

df = pd.DataFrame([
        [2, 1, 0],
        [3, 2,0],
        [3, 4, 0],
        [5, 5, 1],
        [7, 5, 1],
        [2, 5, 0],
        [8, 9, 1],
        [9, 10, 1],
        [6, 12, 1],
        [9, 2, 1],
        [6, 10, 1],
        [2, 4, 0]
    ], columns=['hour', 'attendance', 'pass'])

print(df)
df_x_data = df.drop(['pass'],axis=1)
df_y_data = df['pass']

df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x_data, df_y_data, test_size=0.3, random_state=777, stratify=df_y_data)

print(df_x_train)