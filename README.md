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

2. [데이터 전처리](https://github.com/zoozero127/10th-EDA/tree/main/Team_C/SourceCode/1.%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%88%98%EC%A7%91%26%EC%A0%84%EC%B2%98%EB%A6%AC)

          메타전처리1 : 불필요하다고 판단된 column제거, 회차정보로부터 숫자만 도출, 방영일자로부터 방영요일과 방영시간 도출, 드라마별 장르 원핫 인코딩.  
          전체시청률 : 숫자 형식이 필요한 데이터는 숫자가 나오게끔 변환, 연속회차 방영 여부 적시.  
          searchs&shops : 기준 단어의 검색량을 전체 시계열로 찾고,  [드라마 시작시간과 종료시간] 동안 기준단어와 드라마 비교 검색량 도출.  
                          전체 시계열과 기간 한정 기준 단어 사이의 비례값을 위에서 구한 비교검색량에 곱하여 데이터 간 비교 가능하도록 표준화.  
                          다만 Shops의 경우, log10을 씌운 후 -inf(log 0) 방지를 위해 원래 값에 +0.000001 가산.
          배우브랜드 : 브랜드평판 연구소로부터 웹 크롤링이 가능한 것들은 크롤링하고, 이미지로 제공된 데이터는 OCR 후 오타 검정. 양자 병합 후 yymm 형식 1열에 추가.
   
            
 
4. 분석 방법 및 결과( [시청률 요인](https://github.com/zoozero127/10th-EDA/tree/main/Team_C/SourceCode/2.%EC%8B%9C%EC%B2%AD%EB%A5%A0%EC%97%90%EA%B4%80%ED%95%9C%EB%B6%84%EC%84%9D)  /  [드라마의 영향력](https://github.com/zoozero127/10th-EDA/blob/main/Team_C/SourceCode/3.%EB%84%A4%EC%9D%B4%EB%B2%84api%EC%82%AC%EC%9A%A9%EB%B6%84%EC%84%9D/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%84%EB%B6%84%EC%84%9D%26%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0%EB%A7%81.ipynb) )

         1. 시청률 요인  
         코로나 기간이 시청률에 미치는 영향 분석 : Levene's Test, T Test  
         인기 작품의 장르적 특성 분석 : 시각화  
         인기있는 로맨스 작품을 위한 특성 검출 : UMAP, K-Means  
         드라마 황금시간대 도출 : T Test
         2. 드라마의 영향력  
         드라마 시청률과 네이버 검색어 트렌드 검색량, 네이버 쇼핑인사이트 검색량 상관관계 분석 및 넷플릭스 동시방영 여부가 미치는 영향력 분석  
           * Shapiro Test, Pearson Corr, Spearman Corr, T test, Wilcoxon Test, BoxPlot  
         인기 드라마 특성 검출 : UMAP, K-Means
		    
6. 결론

         코로나 기간동안 시청률이 분명히 늘어났다가, 경보가 해제되면서 점차 낮아진다.
         한국인은 막장, 로맨스 장르 드라마를 좋아한다.
         로맨스 작품은 특히 저녁 7시 드라마가 시청률이 높다.
         황금시간대는 저녁 7시이다. 밤 11시가 넘어서면 시청률이 급감한다.
         OTT는 트렌드 검색량에 유의한 상승 효과를 갖는다. 특히 시청률 상승과 결합하면 드라마 검색량을 큰폭으로 높인다.
         각 클러스터 특성은 다음과 같다.
           * 클러스터 0은 바이럴(검색량)이 높고, 1은 쇼핑검색량이 높으며, 3은 시청률이 높다. 2는 모든 영역이 대체적으로 낮다. 
   ![<Cluster특성>](<https://github.com/zoozero127/10th-EDA/blob/main/Team_C/Dataset/Cluster%ED%8A%B9%EC%84%B1.PNG>)
    
8. 아쉬운 점
    
        1. 업체측 집계 누락으로 배우 브랜드 데이터에 꽤 많은 기간의 데이터가 없음.
        2. 네이버 API를 다루는 시간이 부족해 연령효과 및 성차를 분석할 데이터 수집 실패.
        3. 직관적으로 당연한 결과가 도출되었다는 평가.

10. 추가로 하면 좋을 분석 방법

         연령별, 성차 효과를 고려하여 마케터들이 원하는 타겟에 어필할 수 있는 드라마 장르 특성 검출
         톱배우를 배출할 수 있는 드라마 특성 파악
         제작사나 작가의 명성/경험에 따른 시청률 효과
         전작이 다음 작품 시청률에 미치는 영향

<br>



 ## 각 팀원의 역할
 
|이름|활동 내용| 
|:---:|:---|
|정주영| - (팀장) 전체 일정 관리 및 회의 진행<br> - 크롤링/전처리 과정 보조<br> - 네이버 API 사용데이터 제작<br> - 상관관계 분석 및 클러스터 특성 분석| 
|강건우| - 인기작품의 장르적 특성 분석<br> - 로맨스 클러스터링 및 특성 검출<br> - 황금시간대 분석<br> - 메타데이터 전처리| 
|이준린| - 메타데이터 크롤링 <br> - 발표데이터 총괄 제작<br> - 드라마의 영향력 발표|
|정성오| - 시청률 데이터 크롤링/전처리<br> - 배우브랜드 데이터 범주화<br> - 코로나에 따른 시청률 효과 분석<br> - 중간 발표 <br> - 드라마 시청률 분석 발표| 
<br/>



## tree
```bash
├─Dataset
│      1923_dramalist.csv
│      actor_ranking_(z_score_based)_1901_2306.csv
│      Cluster특성.PNG
│      genre_one_hot_encoding.csv
│      readme.md
│      searchs.csv
│      shops.csv
│      메타_전처리1.csv
│      메타데이터.csv
│      배우브랜드.csv
│      브랜드.xlsx
│      장르만_다시_처리.csv
│      전체시청률.csv
│
├─SourceCode
│  ├─1.데이터수집&전처리
│  │      네이버API_데이터_수집.ipynb
│  │      메타데이터_전처리.ipynb
│  │      메타데이터_크롤링.ipynb
│  │      배우브랜드_전처리.ipynb
│  │      배우브랜드_크롤링.ipynb
│  │      시청률추출&전처리.ipynb
│  │
│  ├─2.시청률에관한분석
│  │      시청률에대한코로나효과.ipynb
│  │      장르효과&로맨스클러스터링&황금시간대.ipynb
│  │
│  └─3.네이버api사용분석
│          상관관계분석&클러스터링.ipynb
│
├─강건우
│      dummy.txt
│      EDA_기초_분석.ipynb
│      EDA_데이터_전처리.ipynb
│      EDA_데이터_전처리2.ipynb
│      드라마_EDA.ipynb
│
├─이준린
│      dummy.txt
│      eda_crawling_metaData.ipynb
│      meta_장르병합_code.ipynb
│
├─정성오
│      actor_rank_(z_score_based).ipynb
│      auto_EDA.ipynb
│      meta_장르병합_code_0729ver.ipynb
│      rating_abt_covid_period_t_test.ipynb
│      test.ipynb
│      드라마_시청률_데이터.csv
│      드라마_장르_원핫인코딩.ipynb
│      시청률_추출.ipynb
│      시청률_추출_못한_드라마_리스트.csv
│
├─정주영
│      dummy.txt
│      EDA_분석시도.ipynb
│      for_cluster.csv
│      searchs.csv
│      shops.csv
│      whole_join.csv
│      드라마_리스트_정제.ipynb
│      배우브랜드.ipynb
│      브랜드.xlsx
│      사전_EDA.ipynb
│      쇼핑분석.ipynb
│      시청률_추출2.ipynb
│
├─ EDA-C조.pdf
└── README.md
``` 
