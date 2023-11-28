# %% [markdown]
# # 데이터 불러오기

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# 기본적인 라이브러리 불러오기
import pandas as pd
import numpy as np
from numpy import datetime64
from datetime import datetime
import math
import os
import unicodedata

# 시각화 라이브러리 불러오기
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from IPython.display import Image
from matplotlib import font_manager, rc
from turtle import color

# 작업 시간 확인 라이브러리 불러오기
from tqdm import tqdm
import time

# warnings 무시
import warnings
warnings.filterwarnings('ignore')

# %%
# 한경글로벌마켓 데이터
df = pd.read_csv('/content/drive/MyDrive/유튜브스크립트/한경글로벌마켓_최종.csv', index_col=0)

# %%
df['ds'] = pd.to_datetime(df['ds'])


# %%
!pip install pykrx
from pykrx import stock

# %%
kosdaq = stock.get_index_ohlcv_by_date('20201006', '20230914', '2001').reset_index() # 2020-10-06부터 2023-09-14까지 코스닥 지수 가져오기 (API)

# %%
kosdaq['날짜'] = pd.to_datetime(kosdaq['날짜'])

# %% [markdown]
# # 데이터 보간

# %%
data = pd.merge(kosdaq, df, left_on='날짜', right_on='ds', how='left')

# %%
# 데이터프레임 정리
data = data.drop(['ds'], axis=1)


# %%
# 데이터 보간
data['title_linear'] = data['title'].interpolate(method='linear')
data['script_linear'] = data['script'].interpolate(method='linear')


# %% [markdown]
# # 시계열 plot (제목)

# %%
data = data.set_index('날짜')


# %% [markdown]
# ### 이동평균

# %%
mov_avg1 = data[['title_linear', '시가', '종가']].rolling(window=5).mean()


# %% [markdown]
# ### 정규화

# %%
# Min-Max Scaling
def min_max_normalize(data):
  min_val = data.min()
  max_val = data.max()
  normalized_data = (data - min_val) / (max_val - min_val)
  return normalized_data

# %%
mov_avg1['title_scaling'] = min_max_normalize(mov_avg1['title_linear'])
mov_avg1['시가_scaling'] = min_max_normalize(mov_avg1['시가'])
mov_avg1['종가_scaling'] = min_max_normalize(mov_avg1['종가'])


# %% [markdown]
# ## (2) 최근 6개월에 대한 분석

# %%
month6_data = data.iloc[-128:]


# %% [markdown]
# ### 이동평균

# %%
mov_avg2 = month6_data[['title_linear', '시가', '종가']].rolling(window=5).mean()

# %% [markdown]
# ### 정규화

# %%
mov_avg2['title_scaling'] = min_max_normalize(mov_avg2['title_linear'])
mov_avg2['시가_scaling'] = min_max_normalize(mov_avg2['시가'])
mov_avg2['종가_scaling'] = min_max_normalize(mov_avg2['종가'])
mov_avg2.head()

# %% [markdown]
# # 시점별 상관분석

# %% [markdown]
# ## 상관분석을 위한 데이터프레임 만들기

# %%
kosdaq = stock.get_index_ohlcv_by_date('20200801', '20230930', '2001').reset_index()


# %%
# 한경글로벌마켓 감성지수 데이터와 코스닥 지수 데이터 merge
data = pd.merge(kosdaq, df, left_on='날짜', right_on='ds', how='left')


# %%
# 필요없는 열 정리
data.drop(['시가', '고가', '저가', '거래량', '거래대금', '상장시가총액', 'unique_id', 'ds'], axis=1, inplace=True)


# %%
# 데이터 보간
data['title_linear'] = data['title'].interpolate(method='linear')
data['script_linear'] = data['script'].interpolate(method='linear')


# %%
# 이동평균 계산
data['title_ma5'] = data['title_linear'].rolling(window=5).mean()
data['script_ma5'] = data['script_linear'].rolling(window=5).mean()
data['종가_ma5'] = data['종가'].rolling(window=5).mean()


# %%
# 종가 slope 계산
data['종가_slope'] = data['종가_ma5'].diff()


# %%
data[data['날짜']=='2020-10-06'] # 한경글로벌마켓 감성지수 데이터 시작 인덱스 42

# %%
data[data['날짜']=='2023-09-14'] # 한경글로벌마켓 감성지수 데이터 끝 인덱스 771

# %% [markdown]
# ## 선행분석/후행분석 코드

# %%
def corr_analysis(mode, df, t, s, f, youtube_col, kosdaq_col):
  # mode : f(선행분석) or b(후행분석)
  # df : 상관분석을 진행할 데이터프레임
  # t : 몇 시점 뒤의 데이터와 상관분석을 할 건지
  # s : df에서 한경글로벌마켓 감성지수 데이터가 시작하는 인덱스
  # f : df에서 한경글로벌마켓 감성지수 데이터가 끝나는 인덱스
  # youtube_col : 상관분석을 진행할 감성지수 데이터의 열 이름(raw data나 ma5)
  # kosdaq_col : 상관분석을 진행할 코스닥 데이터의 열 이름(raw data나 ma5나 slope)

  # 한경글로벌마켓 감성지수 데이터
  hk = df[youtube_col][s:f+1]
  # t시점 전/후의 코스닥 종가 데이터
  if mode=='f':
    kosdaq = df[kosdaq_col][s+t:f+1+t]
  elif mode=='b':
    kosdaq = df[kosdaq_col][s-t:f+1-t]
  else:
    return print('mode를 다시 입력해주세요.')

  # 한경글로벌마켓 감성지수 데이터와 t시점 전/후의 코스닥 종가 데이터 concat
  data_for_corr = pd.concat([hk, kosdaq], axis=1)
  corr_value = data_for_corr.corr().iloc[0,1]
  return corr_value

# %% [markdown]
# ## 전 기간에 대한 분석 (제목)

# %% [markdown]
# ### raw data

# %%
corr_raw = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_raw.loc[i, 't'] = i
  corr_raw.loc[i, '선행분석'] = corr_analysis('f', data, i, 42, 771, 'title_linear', '종가')
  corr_raw.loc[i, '후행분석'] = corr_analysis('b', data, i, 42, 771, 'title_linear', '종가')

corr_raw = corr_raw.set_index('t')

# %% [markdown]
# ### 이동평균

# %%
corr_ma = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_ma.loc[i, 't'] = i
  corr_ma.loc[i, '선행분석'] = corr_analysis('f', data, i, 42, 771, 'title_ma5', '종가_ma5')
  corr_ma.loc[i, '후행분석'] = corr_analysis('b', data, i, 42, 771, 'title_ma5', '종가_ma5')

corr_ma = corr_ma.set_index('t')

# %% [markdown]
# ### 종가 slope와 비교

# %%
corr_slope = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_slope.loc[i, 't'] = i
  corr_slope.loc[i, '선행분석'] = corr_analysis('f', data, i, 42, 771, 'title_ma5', '종가_slope')
  corr_slope.loc[i, '후행분석'] = corr_analysis('b', data, i, 42, 771, 'title_ma5', '종가_slope')

corr_slope = corr_slope.set_index('t')

# %% [markdown]
# ## 2023년 이후 데이터에 대한 분석 (제목)

# %%
data[data['날짜']=='2023-01-02']

# %% [markdown]
# ### raw data

# %%
corr_raw2023 = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_raw2023.loc[i, 't'] = i
  corr_raw2023.loc[i, '선행분석'] = corr_analysis('f', data, i, 596, 771, 'title_linear', '종가')
  corr_raw2023.loc[i, '후행분석'] = corr_analysis('b', data, i, 596, 771, 'title_linear', '종가')

corr_raw2023 = corr_raw2023.set_index('t')

# %% [markdown]
# ### 이동평균

# %%
corr_ma2023 = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_ma2023.loc[i, 't'] = i
  corr_ma2023.loc[i, '선행분석'] = corr_analysis('f', data, i, 596, 771, 'title_ma5', '종가_ma5')
  corr_ma2023.loc[i, '후행분석'] = corr_analysis('b', data, i, 596, 771, 'title_ma5', '종가_ma5')

corr_ma2023 = corr_ma2023.set_index('t')

# %% [markdown]
# ### 종가 slope와 비교

# %%
corr_slope2023 = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_slope2023.loc[i, 't'] = i
  corr_slope2023.loc[i, '선행분석'] = corr_analysis('f', data, i, 596, 771, 'title_ma5', '종가_slope')
  corr_slope2023.loc[i, '후행분석'] = corr_analysis('b', data, i, 596, 771, 'title_ma5', '종가_slope')

corr_slope2023 = corr_slope2023.set_index('t')

# %% [markdown]
# ## 감성지수 slope와 종가 slope 비교 (제목)

# %%
# 감성지수 slope 계산
data['title_slope'] = data['title_ma5'].diff()


# %%
corr_slope_sent2023 = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_slope_sent2023.loc[i, 't'] = i
  corr_slope_sent2023.loc[i, '선행분석'] = corr_analysis('f', data, i, 596, 771, 'title_slope', '종가_slope')
  corr_slope_sent2023.loc[i, '후행분석'] = corr_analysis('b', data, i, 596, 771, 'title_slope', '종가_slope')

corr_slope_sent2023 = corr_slope_sent2023.set_index('t')

# %%
# 한경글로벌마켓 감성지수 5일 이동평균값 & 코스닥 종가 slope 시계열 추이
plt.figure(figsize=(10,6))
fig, ax1 = plt.subplots(1,1)
ax2 = ax1.twinx()
ax1.plot(data['title_slope'][596:771], color='orange')
ax2.plot(data['종가_slope'][596:771], color='blue')

ax1.legend(labels=['한경글로벌마켓 slope'])
ax2.legend(labels=['종가 slope'])

plt.title('한경글로벌마켓 감성지수 slope & 코스닥 종가 slope 비교')
plt.grid(True)

plt.show()

# %% [markdown]
# ## 전 기간에 대한 분석 (스크립트)

# %% [markdown]
# ### raw data

# %%
corr_raw = pd.DataFrame(columns=['t', '선행분석', '후행분석'])
corr_raw.head()

# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_raw.loc[i, 't'] = i
  corr_raw.loc[i, '선행분석'] = corr_analysis('f', data, i, 42, 771, 'script_linear', '종가')
  corr_raw.loc[i, '후행분석'] = corr_analysis('b', data, i, 42, 771, 'script_linear', '종가')

corr_raw = corr_raw.set_index('t')

# %% [markdown]
# ### 이동평균

# %%
corr_ma = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_ma.loc[i, 't'] = i
  corr_ma.loc[i, '선행분석'] = corr_analysis('f', data, i, 42, 771, 'script_ma5', '종가_ma5')
  corr_ma.loc[i, '후행분석'] = corr_analysis('b', data, i, 42, 771, 'script_ma5', '종가_ma5')

corr_ma = corr_ma.set_index('t')

# %% [markdown]
# ### 종가 slope와 비교

# %%
corr_slope = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_slope.loc[i, 't'] = i
  corr_slope.loc[i, '선행분석'] = corr_analysis('f', data, i, 42, 771, 'script_ma5', '종가_slope')
  corr_slope.loc[i, '후행분석'] = corr_analysis('b', data, i, 42, 771, 'script_ma5', '종가_slope')

corr_slope = corr_slope.set_index('t')

# %% [markdown]
# ## 2023년 이후 데이터에 대한 분석 (스크립트)

# %% [markdown]
# ### raw data

# %%
corr_raw2023 = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_raw2023.loc[i, 't'] = i
  corr_raw2023.loc[i, '선행분석'] = corr_analysis('f', data, i, 596, 771, 'script_linear', '종가')
  corr_raw2023.loc[i, '후행분석'] = corr_analysis('b', data, i, 596, 771, 'script_linear', '종가')

corr_raw2023 = corr_raw2023.set_index('t')

# %% [markdown]
# ### 이동평균

# %%
corr_ma2023 = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_ma2023.loc[i, 't'] = i
  corr_ma2023.loc[i, '선행분석'] = corr_analysis('f', data, i, 596, 771, 'script_ma5', '종가_ma5')
  corr_ma2023.loc[i, '후행분석'] = corr_analysis('b', data, i, 596, 771, 'script_ma5', '종가_ma5')

corr_ma2023 = corr_ma2023.set_index('t')

# %% [markdown]
# ### 종가 slope와 비교

# %%
corr_slope2023 = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_slope2023.loc[i, 't'] = i
  corr_slope2023.loc[i, '선행분석'] = corr_analysis('f', data, i, 596, 771, 'script_ma5', '종가_slope')
  corr_slope2023.loc[i, '후행분석'] = corr_analysis('b', data, i, 596, 771, 'script_ma5', '종가_slope')

corr_slope2023 = corr_slope2023.set_index('t')

# %% [markdown]
# ## 감성지수 slope와 종가 slope 비교 (스크립트)

# %%
# 감성지수 slope 계산
data['script_slope'] = data['script_ma5'].diff()


# %%
corr_slope_sent2023 = pd.DataFrame(columns=['t', '선행분석', '후행분석'])


# %%
# 0~29일 전후 시점에 대해 상관분석 진행
for i in range(30):
  corr_slope_sent2023.loc[i, 't'] = i
  corr_slope_sent2023.loc[i, '선행분석'] = corr_analysis('f', data, i, 596, 771, 'script_slope', '종가_slope')
  corr_slope_sent2023.loc[i, '후행분석'] = corr_analysis('b', data, i, 596, 771, 'script_slope', '종가_slope')

corr_slope_sent2023 = corr_slope_sent2023.set_index('t')

# %%
# 한경글로벌마켓 감성지수 5일 이동평균값 & 코스닥 종가 slope 시계열 추이
plt.figure(figsize=(10,6))
fig, ax1 = plt.subplots(1,1)
ax2 = ax1.twinx()
ax1.plot(data['script_slope'][596:771], color='orange')
ax2.plot(data['종가_slope'][596:771], color='blue')

ax1.legend(labels=['한경글로벌마켓 slope'])
ax2.legend(labels=['종가 slope'])

plt.title('한경글로벌마켓 감성지수 slope & 코스닥 종가 slope 비교')
plt.grid(True)

plt.show()

# %% [markdown]
# # 그래프 그리기

# %% [markdown]
# ## 상관분석을 위한 데이터프레임 만들기

# %%
kosdaq = stock.get_index_ohlcv_by_date('20200801', '20230930', '2001').reset_index()


# %%
# 한경글로벌마켓 감성지수 데이터와 코스닥 지수 데이터 merge
data = pd.merge(kosdaq, df, left_on='날짜', right_on='ds', how='left')


# %%
# 필요없는 열 정리
data.drop(['시가', '고가', '저가', '거래량', '거래대금', '상장시가총액', 'unique_id', 'ds'], axis=1, inplace=True)


# %%
# 데이터 보간
data['title_linear'] = data['title'].interpolate(method='linear')
data['script_linear'] = data['script'].interpolate(method='linear')


# %%
# 이동평균 계산
data['title_ma5'] = data['title_linear'].rolling(window=5).mean()
data['script_ma5'] = data['script_linear'].rolling(window=5).mean()
data['종가_ma5'] = data['종가'].rolling(window=5).mean()


# %%
# 종가 slope 계산
data['종가_slope'] = data['종가_ma5'].diff()


# %%
data2023 = data.iloc[596:775,:]


# %%
# 한경글로벌마켓 감성지수 5일 이동평균값 & 코스닥 종가 slope 시계열 추이
fig, axes = plt.subplots(2,1,figsize = (15, 7))
axes[0].plot(data2023['날짜'], data2023['script_ma5'], color='orange')
axes[1].plot(data2023['날짜'], data2023['종가_slope'], color='blue')

axes[0].legend(labels=['한경글로벌마켓'])
axes[1].legend(labels=['종가 slope'])

fig.suptitle('한경글로벌마켓 감성지수 ma5 & 코스닥 종가 slope 비교')
plt.grid(True)

plt.show()

# %%
# 한경글로벌마켓 감성지수 5일 이동평균값 & 코스닥 종가 slope 시계열 추이
plt.figure(figsize=(10,6))
fig, ax1 = plt.subplots(1,1)
ax2 = ax1.twinx()
ax1.plot(data['script_slope'][596:771], color='orange')
ax2.plot(data['종가_slope'][596:771], color='blue')

ax1.legend(labels=['한경글로벌마켓 slope'])
ax2.legend(labels=['종가 slope'])

plt.title('한경글로벌마켓 감성지수 slope & 코스닥 종가 slope 비교')
plt.grid(True)

plt.show()

# %%



