## 👀 데이터 리모델링(Data Remodeling)

사전적 의미: 기존에 설계된 데이터베이스 구조나 데이터 모델을 개선하거나 변경하는 과정

### 데이터 리모델링이 필요한 경우

- 데이터가 많아서 성능이 느려질 때
- 비즈니스 요구사항이 바뀌었을 때
- 중복된 데이터가 많아 문제가 되는 경우
- 데이터의 무결성이 깨질 위험이 있을 때

데이터의 무결성: 데이터의 정확성, 일관성, 유효성이 유지되는 것

### 🤔 데이터 리모델링 VS 데이터 리샘플링

|  | 데이터 리모델링 | 데이터 리샘플링 |
| --- | --- | --- |
| 목적 | 데이터의 구조나 형태를 바꾸는 작업 | 데이터의 시간 간격이나 샘플 수를 바꾸는 작업 |
| 대상 | 데이터베이스, 테이블, 일반 데이터셋 | 시계열 데이터, 시간 기반 데이터 |
| 작업 | 피벗, 병합, 스케일링, 정규화, 리샘플링, 데이터 증강 등 | 시간 단위 변환, 다운샘플링, 업샘플링, 보간법 |
| 파키지 | Pandas, scikit-learn, Dask | Pandas, tslearn, imbalance-learn |

⇒ 데이터 리샘플링은 데이터 리모델링보다 더 하위 개념으로 리모델링의 과정 중 하나이다.

## 📊 데이터 리모델링에 활용 가능한 패키지

### 1. Pandas

1. 데이터 변환
    1. 데이터 병합
    2. 데이터 피벗
    3. 결측 처리
2. 데이터 집계(Aggregation)
    1. 그룹화(Grouping), 집계
    2.  resampling

```python
# 병합 -> 그룹화 -> 집계
grouped_df = pd.merge(df1, df2, on='ID').groupby('Name')['Score'].mean().reset_index()
print(final_df)

```

### 2. NumPy

- 대규모 데이터 변환 행렬 연산, 브로드캐스팅 → 데이터 리모델링 과정이 빠르다.
- 활용: 대규모 수피 데이터 변환 및 리모델링

1. 배열 형태 변환 - `reshape`
2. 배열 평탄화 - `flatten()`, `revel()` : 다차원 배열을 1차원으로 변경
3. 배열 병합 - `concatenate()`, `stack()`, `hstack()`, `vstack()`
    1. vstack, hstack은 딥러닝 시계열 모델 구현을 위한 시퀀스 데이터를 구성하는 데 자주 활용
4. 배열 분할 - `split()`, `hsplit()`, `vsplit()`
5. 데이터 타입 변환 - `astype()` 

### 3. scikit-learn:

- 머신러닝 데이터셋 전처리 군집화, 차원축소 등의 기능을 지원하는 패키지
1. 표준화(Standardization): 데이터의 평균을 0, 표준편차 1로 맞추는 과정
    1. `sklearn.preprocessing` 의 `StandardScaler()` 를 통해 구현
    
    
2. 데이터 정규화: 모든 데이터의 값을 0과 1사이로 맞추는 과정
    1. `sklearn.preprocessing` 의 `MinMaxScaler()` 활용
3. 데이터 인코딩
- 인코딩은 범주형 데이터를 수치형 데이터로 바꾸는 과정
    - `LabelEncoder()` : 명목형(ex. `str`) → `int`
    - `OneHotEncoder()` : pandas의 `get_dummies()`
    - https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

### 4. tslearn

시계열 데이터 리모델링(리샘플링)을 수행할 수 있는 패키지

1. `tslearn.preprocessing.TimeSeriesResampler` 를 활용하여 시계열 데이터 사이즈 변환 수행
    1. 업샘플링, 다운샘플링 둘 다 활용 가능
2. `from tslearn.preprocessing import TimeSeriesScalerMinMax` 를 통해 시계열 데이터 정규화 활용
    1. 데이터 리샘플링 작업 이후에는 데이터의 크기가 달라질 수 있기 때문에 스케일링을 통해 단위를 통일
3. DTW(Dynamic Time Warping) 기반의 리샘플링 수행
    1. DTW 기법은 길이가 다른 두 개의 시계열 데이터 간의 유사도를 측정
    2. DTW 과정에서 길이가 다른 시계열 데이터의 길이를 맞춰줌
    3. `from tslearn.metrics import dtw` 로 구현


---

### 5. imbalance-learn

불균형한 데이터셋을 다룰 때 사용하는 패키지

1. `RandomOverSampler` : 소수 클래스 복제로 데이터 증강
2. **`RandomUnderSampler`**: 다수 클래스 **샘플 제거**로 균형 맞춤.
3. **`SMOTE` (Synthetic Minority Over-sampling Technique)**: **새로운 샘플 생성**으로 증강.
4. **`ADASYN`**: SMOTE와 비슷하지만, **더 복잡한 데이터**에서도 증강.
5. **`SMOTETomek`**: SMOTE + Tomek 링크 제거(중복 제거).


### 💡 한줄 정리

데이터 모델링은 데이터를 분석하기 좋은 형태로 **전체적으로 변환**하는 과정
