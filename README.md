## PatchTST와 KoBert를 활용한 미래 주가 향방 예측
#### 참여자 : 권수현, 박성원, 서연우, 연제명, 정주영 
#### EDA 프로젝트 자료 소개
> * Dataset
>   * 부읽남 : 유튜브 채널 부읽남의 스크립트 크롤링 데이터(크롤링 출처 : [부읽남TV](<https://www.youtube.com/@buiknam_tv>))
>   * 한경글로벌마켓 : 유튜브채널 한국경제의 글로벌마켓 꼭지 스크립트 크롤링 데이터(크롤링 출처 : [한국경제tv](<https://www.youtube.com/@hkwowtv>))
>   * talent : 유튜브채널 달란트투자의 스크립트 크롤링 데이터(크롤링 출처 : [달란트투자](<https://www.youtube.com/@talentinvestment>))
>   * kosdaq_data : 최근 3년간의 Kosdaq 데이터(크롤링 출처 : pykrx 라이브러리 사용)

<br>


## 모델링 프로젝트 요약

1. 프로젝트 주제 및 목적

          실물 경제에 빠르게 대응하는 경제 유튜버의 동영상 스크립트를 통해 지표를 추출하여 미래 주가 향방 예측이 가능한지 확인하고자 하였다.

2. 데이터전처리

          크롤링한 스크립트 데이터의 불용어를 제거하고, KR-FinBert를 통해 긍정어와 중립어로부터 각 채널별 감성지수를 도출하였다.
   
            
 
3. 분석 방법 및 결과

         1. 감성지수와 코스닥 지수 간 상관성 분석
         한경 글로벌 마켓 및 삼프로TV의 경우 유의미, 하지만 부읽남의 상관성은 -0.07로 상당히 저조.  
         2. 시계열분석 결과
         Arima 모형 및 PatchTST 모형을 통한 분석
			- Arima 모형 : 성능 저조
   			- PatchTST 모형
   				* 단변량 : F1 Score 0.71, Accuracy 0.68
   				* 각각의 채널을 통한 분석 : 단변량의 경우보다 성능 저조
   				* 모든 채널을 사용한 경우 : F1 Score 0.71, Accuracy 0.68
   				* 한경/삼프로 채널만 사용 : F1 Score 0.74, Accuracy 0.8
   

4. 결론

         PatchTST 모형은 단변량으로 사용한 경우도 어느정도 유효함.
         모든 채널을 사용한 경우보다 이전에 상관성이 높은 것으로 나타난 한경 글로벌 마켓 및 삼프로TV를 사용하여 예측을 시도한 경우 유의미하게 성능 지표가 향상된 것을 확인.
         코스닥에 존재하는 유튜버의 영향력을 간접적으로 확인.
         주간 또는 월간 지수 향방 예측 가능성 모색. 
         실제 투자 결과, 수익 성공.
    
5. 아쉬운 점
    
        1. 부족한 데이터. 더 많은 채널을 고려할 수도 있음.
        2. 스크립트 처리의 한계 : 유튜버 특성상 경제와 무관한 대화가 지나차게 많은 경우도 있음.
        3. 개별 주가 예측에는 실패.


<br>



 ## 각 팀원의 역할
 
|이름|활동 내용| 
|:---:|:---|
|정주영| - (팀장) 전체 일정 관리 및 회의 진행<br> - PatchTST모형 실행<br> - 문제 해결 아이디어 제공|
|권수현| - 크롤링 코드 작성<br> - 상관관계 분석<br> - 데이터 전처리<br> - NLP모델 활용<br> - NLP모델 문제 해결|
|박성원| - 크롤링 실행<br> - 상관관계 분석<br> - 데이터전처리<br> - NLP모델 활용<br>|
|서연우| - PatchTST모형 이론적 배경 제공<br> - 각 파이프라인 간 유기적 연결 <br> - 토의 진행<br> - 발표자료 작성 및 발표|
|연제명| - 주제 및 진행방향 제시<br> -  상관관계 분석<br> - 데이터 전처리, NLP모델 제시|
<br/>



## tree
```bash
├─Dataset
│      3pro_sent.csv
│      kosdaq_data.csv
│      talent_sent.csv
│      부읽남_최종.csv
│      한경글로벌마켓_최종.csv
│
├─SourceCode
│  ├─NLP_Preprocessing
│  │      3pro.py
│  │      data_preprocessing.py
│  │      talent_data_preprocessing.py
│  │
│  ├─Correlation
│  │      부읽남.py
│  │      삼프로.py
│  │	  한경.py
│  │
│  └─Timeseries_Forecasting
│  │      Arima.py
│  │      PatchTST.py
│  │      Arima&PatchTST_h_50.py
│  │
├─ 유감TV_KOSDAQ.pdf
└── README.md
``` 
