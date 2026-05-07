# 농산물 가격 예측 ML 프로젝트

농산물 유통 데이터를 활용해 주요 품목 7개 종의 **미래 가격을 예측하는 머신러닝 모델**을 구축하고, 예측 결과를 **Streamlit 대시보드**로 배포한 데이터 분석 & 머신러닝 프로젝트입니다.

EDA, 시계열 피처 엔지니어링, Prophet 및 ML/앙상블 모델 비교, SHAP 해석, Streamlit 배포까지 연결하여 구성했습니다.


## What I Demonstrated In This Project

- 실제 공공/유통 데이터를 활용한 문제 정의 능력
- 시계열 데이터 EDA 및 전처리 역량
- ML 회귀/앙상블 모델 실험 및 비교 역량
- SHAP 기반 모델 해석 역량
- Streamlit 기반 데이터 제품화 경험
- 비즈니스 관점에서 분석 결과를 해석하는 능력


![웹사이트 화면 예시](./images/website_screenshot.png)

*웹사이트 화면 예시 (url: https://prediction-ml-ppck75.streamlit.app/)*


## 1. Project Overview

### 문제 정의
- 농산물 가격은 계절성, 휴일, 수급 불안, 외부 이벤트의 영향을 크게 받습니다.
- 가격 변동성이 큰 품목은 매입 시점과 물량 판단이 어렵고, 이는 재고 비용과 폐기 비용으로 이어질 수 있습니다.
- 따라서 과거 거래 데이터를 바탕으로 **품목별 가격을 예측**해, 실무에서 활용 가능한 의사결정 보조 지표를 만드는 것을 목표로 했습니다.

### 비즈니스 연관성
- 마켓컬리와 같은 신선식품 이커머스 기업은 **직매입 구조**를 통해 상품을 조달하고 직접 물류 부담을 집니다.
- 이 구조에서는 가격 급등락에 따라 매입 원가, 재고 리스크, 물류 운영 효율이 크게 달라집니다.
- 본 프로젝트의 가격 예측 모델은 다음과 같은 의사결정과 연결될 수 있습니다.
  > 적정 매입 시점 판단  
  > 수급 변동에 따른 선제적 재고 운영  
  > 가격 급등 가능성에 대한 리스크 관리  
  > 프로모션 및 판매 정책 수립 지원  
  
(출처: 메타코드 Liam 강사님 머신러닝 강의자료)

## 2. Dataset

### 데이터 출처
- [농넷 | 농산물유통종합정보시스템](https://www.nongnet.or.kr/index.do)

### 데이터 설명
- 2016-01-01부터 약 4년 이상 누적된 농산물 유통 데이터 사용
- 품목별 **거래량**과 **가격(원/kg)** 정보 포함
- 본 프로젝트에서는 가격 예측에 집중하기 위해 **타깃을 가격 변수로 한정**

### 예측 대상 품목
- 배추
- 무
- 마늘
- 양파
- 대파
- 건고추
- 깻잎

이 품목들은 수급 불안정성이 크고 가격 민감도가 높아, 실제 유통/리테일 도메인에서 예측 가치가 높다고 판단했습니다.

## 3. Objectives

- 시계열 데이터 특성을 반영한 EDA 수행
- 날짜 기반/시차 기반 feature engineering 설계
- Prophet과 머신러닝 회귀 모델 성능 비교
- 후처리 및 앙상블을 통한 예측 성능 개선
- 결과를 Streamlit으로 시각화하여 대시보드 형태로 배포 (url: https://prediction-ml-ppck75.streamlit.app/)

## 4. Project Workflow

*본 프로젝트는 메타코드 Liam 강사님의 kaggle_실전 머신러닝 강의를 참고하여 구성했습니다.*

### 1) EDA
- 날짜 누락 여부 확인 및 시계열 정렬
- 품목별 가격 추이 시각화
- 이동평균(rolling mean)으로 장기 트렌드 확인
- 요일/월 단위 가격 패턴 분석
- 품목 간 상관관계 확인
- 휴일 및 특정 시즌 전후 가격 움직임 탐색

### 2) Feature Engineering
노트북 기반 모델링에서는 시계열 예측 성능 개선을 위해 다음과 같은 피처를 설계했습니다.

- 날짜 파생 변수
  - `year`, `month`, `day`, `day_of_week`
  - `is_weekend`, `is_holiday`
- 시차 변수
  - `lag_7`, `lag_14`
- 이동 통계 변수
  - rolling mean
  - rolling std

핵심은 단순 가격 예측이 아니라, **과거 시점의 패턴과 달력 정보를 수치화해 모델이 학습할 수 있게 만든 것**입니다.

### 3) Modeling

#### Baseline
- Prophet

#### Machine Learning Regressors
- Ridge Regression
- Random Forest Regressor
- LightGBM Regressor
- XGBoost Regressor
- MLP Regressor

#### Ensemble
- Average Ensemble
- Stacking Regressor

### 4) Post-processing
- 일요일/공휴일과 같이 거래 특성이 비정상적인 날짜에 대해 예측값을 보정
- 노트북 실험 기준으로 후처리 적용 시 일부 품목에서 성능 개선 확인

이 단계는 단순히 모델 점수를 높이기 위한 트릭이 아니라, **도메인 규칙을 예측 결과에 반영하는 실무적 접근**이라는 점에서 의미가 있습니다.

## 5. Evaluation

### 평가 지표
- `MdAPE (Median Absolute Percentage Error Accuracy 관점으로 해석)`

노트북에서는 품목별 가격 스케일 차이가 크기 때문에, 절대 오차보다 **상대 오차 중심의 비교**가 적절하다고 판단해 MdAPE 기반으로 모델 성능을 평가했습니다.

### 실험 요약
- 단일 모델만으로도 품목에 따라 강점이 달랐습니다.
- 일부 품목은 XGBoost/RandomForest 계열이 강했고, 일부는 평균 앙상블이 더 안정적인 성능을 보였습니다.
- 후처리 적용 후 전반적으로 예측 품질이 개선되는 패턴을 확인했습니다.
- 노트북 실험 결과 기준으로 **품목별 최적 모델이 달라질 수 있음**을 확인했고, 이는 실제 서비스에서 품목별 모델 선택 전략이 필요함을 시사합니다.

## 6. Interpretability

- SHAP 기반 feature importance를 통해 예측에 영향을 크게 주는 변수 확인
- 과거 가격(lag), 이동평균, 달력성 변수의 중요도를 해석
- 단순히 "잘 맞는 모델"이 아니라, **무엇이 가격 변동에 영향을 주는지 설명 가능한 모델링**을 지향

## 7. Deployment

### Streamlit Dashboard
- 모델 예측 결과와 실제 가격 흐름을 함께 시각화
- 품목별 예측 결과 비교
- 모델별 성능 요약표 제공  
> 사전에 생성된 농산물 가격 예측 데이터를 기반으로, 사용자가 선택한 품목과 기간에 따른 가격 추이를 시각화하고, 이동평균선과 모델 성능 요약을 표시합니다.
> 학습한 ML 모델의 예측 데이터를 불러와서 품목별 가격 추이와 이동평균선을 시각화

## 8. Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Prophet
- Scikit-learn
- LightGBM
- XGBoost
- SHAP
- Streamlit

## 9. Installation

### App 실행 환경

Streamlit 앱만 실행할 경우에는 루트 `requirements.txt`만 설치하면 됩니다.

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Notebook 실험 환경

모델 학습, EDA, 앙상블 실험까지 재현하려면 `notebooks/requirements.txt`를 설치합니다.

```bash
pip install -r notebooks/requirements.txt
```

노트북 파일은 아래 경로를 기준으로 관리하는 것을 권장합니다.

```bash
notebooks/agricultural_price_forecasting.ipynb
```

## 10. Key Takeaways

- 시계열 문제에서는 단순 회귀보다 **시간 순서와 누수(leakage) 방지**가 훨씬 중요하다는 점을 학습했습니다.
- feature engineering이 모델 종류 자체보다 더 큰 성능 차이를 만들 수 있음을 확인했습니다.
- 도메인 지식이 반영된 후처리가 실제 예측 성능 개선에 기여했습니다.
- 분석 결과를 Streamlit으로 배포하면서, 기술 구현뿐 아니라 **결과 전달 방식**도 프로젝트 완성도에 중요하다는 점을 경험했습니다.


## 11. Future Improvements

- 날씨, 기온, 강수량, 명절 수요 등 외부 변수 추가
- 품목별 개별 최적화 모델 운영
- 시계열 교차검증(Time Series Cross Validation) 고도화
- 예측 구간 확장 및 다중 시점 forecasting
- 운영 환경 기준의 자동 배치 예측 파이프라인 구축

## 12. Reference  

Liam 강사 / kaggle_실전_머신러닝_강의자료(메타코드)

*좋은 강의로 학습과 프로젝트 제작에 도움을 주신 Liam 강사님께 감사드립니다*

---


