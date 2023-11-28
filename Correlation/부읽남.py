# %%
%%capture
!pip install neuralforecast
!pip install datasetsforecast
!pip install pytorch_lightning
!pip install nbdev
!pip install pykrx

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykrx import stock
from tqdm import tqdm
import math
# https://github.com/Nixtla/neuralforecast/tree/main/
# https://github.com/Nixtla/neuralforecast/blob/main/nbs/examples/Forecasting_TFT.ipynb


# %%
kosdaq = stock.get_index_ohlcv_by_date('20200909', '20230922', '2001').reset_index()
df = kosdaq.copy()[['날짜', '종가']]
df['unique_id'] = 'kosdaq'
df.columns = ['ds', 'y', 'unique_id']
df = df[['unique_id', 'ds', 'y']]

# %%
kosdaq = stock.get_index_ohlcv_by_date('20200913', '20230912', '2001').reset_index()
kosdaq = kosdaq.copy()[['날짜', '종가']]
kosdaq['unique_id'] = 'kosdaq'
kosdaq.columns = ['ds', 'y', 'unique_id']
kosdaq = kosdaq[['unique_id', 'ds', 'y']]

# %%
pro3 = pd.read_csv('/content/drive/MyDrive/유튜브스크립트/부읽남_최종.csv', parse_dates = ['ds'], index_col = 0).sort_values('ds').reset_index(drop = True)
pro3.columns = ['unique_id', 'ds', 'y','title_y']

# %% [markdown]
# # 시점 반영 데이터
# 

# %%
data = pd.merge(kosdaq, pro3, on='ds', how='left')

# %%
data['y_y'] = data['y_y'].interpolate(method='linear')

# %%
data['y_x_ma5'] = data['y_x'].rolling(window=5).mean()
data['y_y_ma5'] = data['y_y'].rolling(window=5).mean()

# %%
pro3_data = data[['unique_id_y', 'y_y', 'y_y_ma5']][6:1067].reset_index(drop=True)

# %%
kosdaq_data = data[['ds', 'unique_id_x', 'y_x', 'y_x_ma5']][21:].reset_index(drop=True) # 삼프로 일별 감성지수 15일 뒤 코스닥 데이터

# %%
time_df = pd.concat([kosdaq_data, pro3_data], axis=1)

# %%
kosdaq_df = time_df.copy()[['unique_id_x', 'ds', 'y_x', 'y_x_ma5']]
kosdaq_df.columns = ['unique_id', 'ds', 'y', 'y_ma5']
kosdaq_df.dropna(subset=['ds'], inplace=True)

# %%
# 삼프로 감성지수 데이터
pro3_df = time_df.copy()[['unique_id_y', 'ds', 'y_y', 'y_y_ma5']]
pro3_df.columns = ['unique_id', 'ds', 'y', 'y_ma5']
pro3_df.dropna(subset=['ds'], inplace=True)
pro3_df['unique_id'] = '3pro'

# %%
df1 = kosdaq[0:].reset_index(drop=True) # 전 기간
df2 = kosdaq[596:].reset_index(drop=True) # 2023년 이후

# %%
# 단일 시계열 - 이동평균
df3 = df1.copy()
df3['y'] = df3['y'].rolling(window=5).mean() # 전 기간
df3.dropna(subset=['y'], inplace=True)

df4 = df2.copy()
df4['y'] = df4['y'].rolling(window=5).mean() # 2023년 이후
df4.dropna(subset=['y'], inplace=True)

# 삼프로 제목 감성지수 사용(시점 반영x)
pro3_1 = data.copy()[['unique_id_y', 'ds', 'y_y', 'y_y_ma5']][26:768].reset_index(drop=True)
pro3_1.columns = ['unique_id', 'ds', 'y', 'y_ma5']
pro3_1['unique_id'] = '3pro'

df5 = pd.concat([df1, pro3_1.iloc[:,:3]]).reset_index(drop=True) # 전 기간
df6 = pd.concat([df2, pro3_1.iloc[570:,:3]]).reset_index(drop=True) # 2023년 이후

# 삼프로 제목 감성지수 사용(시점 반영x) - 이동평균
pro3_2 = pro3_1.copy().iloc[:,[0,1,3]]
pro3_2.columns = ['unique_id', 'ds', 'y']

df7 = pd.concat([df3, pro3_2]).reset_index(drop=True) # 전 기간
df7.dropna(subset=['y'], inplace=True)
df8 = pd.concat([df4, pro3_2.iloc[570:,:]]).reset_index(drop=True) # 2023년 이후

# 삼프로 제목 감성지수 사용(시점 반영o)
df9 = pd.concat([kosdaq_df.iloc[:,:3], pro3_df.iloc[:,:3]]).reset_index(drop=True) # 전 기간
df10 = pd.concat([kosdaq_df.iloc[555:,:3], pro3_df.iloc[555:,:3]]).reset_index(drop=True) # 2023년 이후

# 삼프로 제목 감성지수 사용(시점 반영o) - 이동평균
kosdaq_1 = kosdaq_df.copy().iloc[:,[0,1,3]]
kosdaq_1.columns = ['unique_id', 'ds', 'y']
pro3_3 = pro3_df.copy().iloc[:,[0,1,3]]
pro3_3.columns = ['unique_id', 'ds', 'y']

df11 = pd.concat([kosdaq_1, pro3_3]).reset_index(drop=True) # 전 기간
df11.dropna(subset=['y'], inplace=True)
df12 = pd.concat([kosdaq_1.iloc[555:,:], pro3_3.iloc[555:,:]]).reset_index(drop=True) # 2023년 이후


