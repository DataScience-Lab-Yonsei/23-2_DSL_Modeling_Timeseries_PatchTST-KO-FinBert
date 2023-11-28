%%capture
!pip install neuralforecast
!pip install datasetsforecast
!pip install pytorch_lightning
!pip install nbdev
!pip install pykrx
!pip install pmdarima


import torch
import torch.nn as nn
import torch.nn.functional as F
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



# Horizon = 50
val_size = 50
test_size = 50
horizon = test_size

train_df = df.iloc[:(len(df) - test_size), :3]
test_df = df.iloc[(len(df) - test_size):, :3]

val_time = df['ds'][len(df) - val_size - test_size]
test_time = df['ds'][len(df) - test_size]

final_obs = df.iloc[len(df) - test_size -1, 2]
## AutoArima

model = pm.auto_arima(df['y'],
                      seasonal=True,
                      stepwise=True,
                      suppress_warnings=True,
                      error_action="ignore",
                      max_order=None,
                      trace=True)

forecast = model.predict(n_periods=horizon)


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
## 단일 시계열
train_df = df.loc[df['ds'] < test_time]
test_df = df.loc[df['ds'] >= test_time]
model = PatchTST(h=horizon,
                 input_size=210,
                 patch_len=10,
                 stride=4,
                 revin=True,
                 hidden_size=48,
                 n_heads=48,
                 scaler_type='standard',
                 #loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
                 loss=MAE(),
                 learning_rate=1e-5,
                 max_steps=2000,
                 val_check_steps=50,
                 early_stop_patience_steps=5,
                 activation = 'relu',
                 batch_normalization = True,
                 batch_size = 32,
                 random_seed = 1)

nf = NeuralForecast(models=[model], freq='B')

nf.fit(df=train_df,
      #  static_df=static_df,
       val_size=val_size)
forecasts = nf.predict()

hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([test_df.reset_index(drop = True), hat_df], axis=1)

prac = plot_df.loc[plot_df['unique_id'] == 'kosdaq']
prac['lag'] = 0
prac['lag'] = prac['PatchTST'].shift(1)
prac = prac.fillna(final_obs)
# # 미분 X일 때
# prac['updown_pred'] = prac.apply(lambda row: 1 if row['PatchTST'] > row['lag'] else 0, axis = 1)
# 미분했을 때
prac['updown_pred'] = prac.apply(lambda row: 1 if row['PatchTST'] > 0 else 0, axis = 1)

y_true = df2.iloc[-horizon:, :]['updown_true'].tolist()
y = test_df.loc[test_df['unique_id'] == 'kosdaq', 'y'].tolist()
y_pred = prac['updown_pred'].tolist()
pred = prac['PatchTST'].tolist()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mse = mean_squared_error(y, pred)

print(f'평가지표 : [Accuracy :{accuracy}], [F1 Score : {f1}], [MSE : {mse}]')
plt.plot('ds', 'PatchTST', data = plot_df.loc[plot_df['unique_id'] == 'kosdaq'], color = 'red')
plt.plot('ds', 'y', data = plot_df.loc[plot_df['unique_id'] == 'kosdaq'], color = 'black')
plt.title('단일 시계열')

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))
plt.axhline(y=0, color='k', linestyle='--')
plt.legend()
plt.show()
## 전부 사용
#전부 사용
concat_df = pd.concat([df.iloc[:,:3], reader_df_y, reader_df_title, han_df_y, han_df_title, pro3_df, talent_df]).reset_index(drop = True)
train_df = concat_df.loc[concat_df['ds'] < test_time]
test_df = concat_df.loc[concat_df['ds'] >= test_time]

static_df = pd.DataFrame({'unique_id' : ['kosdaq', f'{word1}', f'{word2}', f'{word3}', f'{word4}', f'{word5}', f'{word6}'],
                         'kosdaq' : [1, 1, 1, 1, 1, 1, 1],
                         f'{word1}' : [1, 0, 1, 1, 1, 1, 1],
                         f'{word2}' : [1, 1, 0, 1, 1, 1, 1],
                         f'{word3}' : [1, 1, 1, 0, 1, 1, 1],
                         f'{word4}' : [1, 1, 1, 1, 0, 1, 1],
                         f'{word5}' : [1, 1, 1, 1, 1, 1, 1],
                         f'{word6}' : [1, 1, 1, 1, 1, 1, 1]
                         })
model = PatchTST(h=horizon,
                 input_size=210,
                 patch_len=10,
                 stride=4,
                 revin=True,
                 hidden_size=48,
                 n_heads=48,
                 scaler_type='standard',
                 #loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
                 loss=MAE(),
                 learning_rate=1e-5,
                 max_steps=2000,
                 val_check_steps=50,
                 early_stop_patience_steps=5,
                 activation = 'relu',
                 batch_normalization = True,
                 batch_size = 32,
                 random_seed = 1)

nf = NeuralForecast(models=[model], freq='B')

nf.fit(df=train_df,
       static_df=static_df,
       val_size=val_size)
forecasts = nf.predict()

hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([test_df.reset_index(drop = True), hat_df], axis=1)

prac = plot_df.loc[plot_df['unique_id'] == 'kosdaq']
prac['lag'] = 0
prac['lag'] = prac['PatchTST'].shift(1)
prac = prac.fillna(final_obs)
# # 미분 X일 때
# prac['updown_pred'] = prac.apply(lambda row: 1 if row['PatchTST'] > row['lag'] else 0, axis = 1)
# 미분했을 때
prac['updown_pred'] = prac.apply(lambda row: 1 if row['PatchTST'] > 0 else 0, axis = 1)

y_true = df2.iloc[-horizon:, :]['updown_true'].tolist()
y = test_df.loc[test_df['unique_id'] == 'kosdaq', 'y'].tolist()
y_pred = prac['updown_pred'].tolist()
pred = prac['PatchTST'].tolist()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mse = mean_squared_error(y, pred)

print(f'평가지표 : [Accuracy :{accuracy}], [F1 Score : {f1}], [MSE : {mse}]')
plt.plot('ds', 'PatchTST', data = plot_df.loc[plot_df['unique_id'] == 'kosdaq'], color = 'red')
plt.plot('ds', 'y', data = plot_df.loc[plot_df['unique_id'] == 'kosdaq'], color = 'black')
plt.title('전부 사용')

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))
plt.axhline(y=0, color='k', linestyle='--')
plt.legend()
plt.show()
## 한경&삼프로
# 한경_스크립트&삼프로
concat_df = pd.concat([df.iloc[:, :3], han_df_y, pro3_df]).reset_index(drop = True)
train_df = concat_df.loc[concat_df['ds'] < test_time]
test_df = concat_df.loc[concat_df['ds'] >= test_time]

static_df = pd.DataFrame({'unique_id' : ['kosdaq', f'{word4}', f'{word5}'],
                          'kosdaq' : [1, 1, 1],
                          f'{word4}' : [1, 1,1],
                          f'{word5}' : [1, 1, 1]
                          })
model = PatchTST(h=horizon,
                 input_size=210,
                 patch_len=10,
                 stride=4,
                 revin=True,
                 hidden_size=48,
                 n_heads=48,
                 scaler_type='standard',
                 #loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
                 loss=MAE(),
                 learning_rate=1e-5,
                 max_steps=2000,
                 val_check_steps=50,
                 early_stop_patience_steps=5,
                 activation = 'relu',
                 batch_normalization = True,
                 batch_size = 32,
                 random_seed = 1)

nf = NeuralForecast(models=[model], freq='B')

nf.fit(df=train_df,
       static_df=static_df,
       val_size=val_size)
forecasts = nf.predict()

hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([test_df.reset_index(drop = True), hat_df], axis=1)

prac = plot_df.loc[plot_df['unique_id'] == 'kosdaq']
prac['lag'] = 0
prac['lag'] = prac['PatchTST'].shift(1)
prac = prac.fillna(final_obs)
# # 미분 X일 때
# prac['updown_pred'] = prac.apply(lambda row: 1 if row['PatchTST'] > row['lag'] else 0, axis = 1)
# 미분했을 때
prac['updown_pred'] = prac.apply(lambda row: 1 if row['PatchTST'] > 0 else 0, axis = 1)

y_true = df2.iloc[-horizon:, :]['updown_true'].tolist()
y = test_df.loc[test_df['unique_id'] == 'kosdaq', 'y'].tolist()
y_pred = prac['updown_pred'].tolist()
pred = prac['PatchTST'].tolist()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mse = mean_squared_error(y, pred)

print(f'평가지표 : [Accuracy :{accuracy}], [F1 Score : {f1}], [MSE : {mse}]')
plt.plot('ds', 'PatchTST', data = plot_df.loc[plot_df['unique_id'] == 'kosdaq'], color = 'red')
plt.plot('ds', 'y', data = plot_df.loc[plot_df['unique_id'] == 'kosdaq'], color = 'black')
plt.title('한경 스크립트&삼프로 제목')

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))
plt.axhline(y=0, color='k', linestyle='--')
plt.legend()
plt.show()