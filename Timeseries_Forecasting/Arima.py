# %%capture
# !pip install neuralforecast
# !pip install datasetsforecast
# !pip install pytorch_lightning
# !pip install nbdev
!pip install pykrx
!pip install pmdarima
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykrx import stock
from tqdm import tqdm
import math
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
import pmdarima as pm


kosdaq = stock.get_index_ohlcv_by_date('20200909', '20230922', '2001').reset_index()
df = kosdaq.copy()[['날짜', '종가']]
df['unique_id'] = 'kosdaq'
df.columns = ['ds', 'y', 'unique_id']
df = df[['unique_id', 'ds', 'y']]
#미분
df['y'] = np.gradient(df['y'], 1)

df2 = kosdaq.copy()
df2['lag'] = df2['종가'].shift(1)
df2['승강'] = 0
df2 = df2.iloc[1:,:]
df2['승강'] = df2.apply(lambda row: 1 if row['종가'] > row['lag'] else 0, axis=1) # 1이면 상승
df2 = df2[['날짜', '승강']]
df2.columns = ['ds', 'updown_true']
df2['ds'] = pd.to_datetime(df2['ds'])

val_size = 50
test_size = 25
horizon = test_size

train_df = df.iloc[:(len(df) - test_size), :3]
test_df = df.iloc[(len(df) - test_size):, :3]

val_time = df['ds'][len(df) - val_size - test_size]
test_time = df['ds'][len(df) - test_size]

final_obs = df.iloc[len(df) - test_size -1, 2]
# AutoArima

model = pm.auto_arima(df['y'],
                      seasonal=True,
                      stepwise=True,
                      suppress_warnings=True,
                      error_action="ignore",
                      max_order=None,
                      trace=True)

forecast = model.predict(n_periods=25)


prac = forecast.reset_index()
prac.columns = ['index', 'Arima']
prac['lag'] = 0
prac['lag'] = prac['Arima'].shift(1)
prac = prac.fillna(final_obs)
# # 미분 X일 때
# prac['updown_pred'] = prac.apply(lambda row: 1 if row['PatchTST'] > row['lag'] else 0, axis = 1)
# 미분했을 때
prac['updown_pred'] = prac.apply(lambda row: 1 if row['Arima'] > 0 else 0, axis = 1)
prac['ds'] = df2.iloc[-horizon:, 0].reset_index(drop = True)

y_true = df2.iloc[-horizon:, :]['updown_true'].tolist()
y = test_df.loc[test_df['unique_id'] == 'kosdaq', 'y'].tolist()
y_pred = prac['updown_pred'].tolist()
pred = prac['Arima'].tolist()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mse = mean_squared_error(y, pred)

print(f'평가지표 : [Accuracy :{accuracy}], [F1 Score : {f1}], [MSE : {mse}]')


plt.plot('ds', 'Arima', data = prac, color = 'red')
plt.plot('ds', 'y', data = df.iloc[-horizon:], color = 'black')
plt.title('Arima(5, 0, 0)(0,0,0)[0]')

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))
plt.axhline(y=0, color='k', linestyle='--')
plt.legend()
plt.show()