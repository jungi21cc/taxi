# Kaggle : New York City Trip Duration

### 개요
- NYC Taxi의 데이터를 사용하여 뉴욕시의 택시여행시간을 예측
- RMSLE를 통한 분류모델 평가
- Light GBM 모델 사용
“최종 결과 : ​ 0.38180​ (상위 16% )”

### 데이터

Train.csv
- 트레이닝 데이터 : ​ (1458644, 11)
- Target 데이터 : trip duration (seconds)
- Target 데이터의 스케일링때문에 log로 변환후 사용

Test.csv테스트 데이터 : (625134, 9)
- 트레이닝 데이터를 통한 Target 데이터 예측
- 트레이닝 데이터중 dropoff time과 pickup time의 차이가 target data dropoff date time 혹은 trip duration을 예측

### 데이터 엔지니어링

Null 데이터 비율
- 없음

Outlier 제거
- 위도, 경도를 기준으로 퀘벡, 센프란시스코등 위치 이상값 제거
- 100,000초가 넘는 데이터 제거
- 총 4개의 데이터 제거

시간 데이터 처리
- 년, 월, 일, 시간, 요일 사용
- 주말데이터 추가 사용
- 미국 국가공휴일을 web crawling을 통해 사용
- 시간대별 trip duration의 분포가 다름을 통해 office hour 사용

거리 데이터
- Dropoff 와 pickup의 차이인 Euclidean, manhattan distance사용
- direction 이라는 방향 데이터 사용 (arc tangent 이용)

위도경도 데이터
- 전통적인 공간데이터 처리시 PCA 및 Linear Discriminant Analysis를 사용
- pick-up, drop-off의 위도, 경도 를 PCA로 변환(2D 에서 2D로 변환)
- 좌표데이터의 noise만 제거하고 정보손실 최소화

클러스터링
- Pick-up, drop-off의 위도, 경도에 해당하는 clustering 실행
- DBSCAN, Spectral Clustering의 경우 메모리 부족현상으로 실행불가(32G메모리)
- 성능이 떨어지지만 K-means, Gaussian Mixture를 사용해 좌표 clustering 사용
- 교차검증기준 k-means보다 Gaussian mixture중 20개의 군집 선택

카테고리 데이터
- 카테고리 변수를 더미변수로 변경

“최종 24개 Feature 사용”

모델링교차검증
- RMSE로 분류검증
- OLS, Ridge Regression, Randomforest, Light GBM, Ensemble model등을 활용한 성능 평가

“교차검증기준 Light GBM이 구동시간 및 성능이 가장 우수”

- Light GBM 파라미터 튜닝 (교차검증기준 가장 성능이 우수한 parameter 선택)
Boost-type = ‘gbtr’
Evaluation metric = rmse
Learning rate = 0.1
Number of estimator = 15,000
L1 regularization = 0.5
L2 regularization = 0.5

### 결론

RMSLE결과: 0.38174 (상위 16%)

"
트레이닝데이터로부터 구해진 속도를 위치 클러스터링에 적용해, 속도를 반영한
클러스터링을 시도해볼 경우 성능개선 요지
다소 높은 다중공정성을 보이나 트레이닝데이터의 feature수가 소수의 관계로
feature engineering시의 다중공정성 문제는 불가피
"
