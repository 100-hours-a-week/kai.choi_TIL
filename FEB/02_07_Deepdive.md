<aside>
💡데이터 분석: 유용한 정보를 발견하고, 결론을 내고, 의사 결정을 지원하기 위해 데이터를 탐색, 처리, 변환, 모델링의 과정
</aside>

→ 데이터로 부터 유용한 인사이트를 얻기 위해서는 원하는 형태 혹은 정보만을 추출할 수 있는 데이터 전처리가 중요하다.

## 데이터 전처리 유형

- Cleansing
- Normalization
- Data Augmentation
- Transform
- Feature engineering
- Formatting ← merge, pivot

이 중, merge와 pivot은 주어진 데이터 형태를 변환하는 Formatting 작업이다.  데이터 분석가의 의도에 따라 원하는 인사이트를 얻기 위해 원하는 정보가 직관적으로 확인 가능하도록 하는 작업이다. 하지만, merge와 pivot 모두 사용자의 데이터 분석 역량에 따라 결과물의 질이 달라질 수 있다.

## 병합(Merge)

https://en.wikipedia.org/wiki/Wikipedia:Merging

## Reasons for merging in Wikipedia

| **Overlap(중복)** | 정확히 동일한 주제에 대한 페이지(데이터 프레임)가 두 개 이상 있고 범위가 동일한 경우. |
| --- | --- |
| **Duplicate(중복)** | 정확히 동일한 주제에 대한 페이지(데이터 프레임)가 두 개 이상 있고 범위가 동일한 경우. |
| **Context(문맥적 용이성)** | 필요한 정보가 존재하는 데이터 프레임의 공통점을 통계치를 도출. |
| **Short text** | 광범위한 자료를 필요한 부분만 추출하여 가독성을 높임. |
| **Insufficient notability** | 일부 컬럼(feature)의 경우 불필요한 정보가 들어있는 경우가 있다. 병합 과정을 통해 불필요한 정보를 삭제할 수 있음. |

## Merge 방식(SQL)

- pandas.merge의 how 인자에 조건을 기입
    - left: **왼쪽 데이터프레임의 모든 데이터를 유지**하고, **오른쪽에서 일치하는 값만 가져옴**
    - right: **오른쪽 데이터프레임의 모든 데이터를 유지**하고, **왼쪽에서 일치하는 값만 가져옴**
    - outer: 양쪽 데이터프레임의 모든 데이터를 유지
    - inner: 공통된 키 값이 존재하는 데이터만 병합
    - cross: 두 개의 데이터프레임에서 모든 가능한 조합을 생성하는 병합

## 활용 예시

```python
# DataFrames
df_sales = pd.DataFrame({'Product': ['A', 'B', 'C'], 'Sales': [100, 150, 200]})
df_price = pd.DataFrame({'Product': ['A', 'B', 'D'], 'Price': [10, 15, 25]})

# Merge
df_merged = df_sales.merge(df_price, on='Product', how='inner')
print(df_merged)
```

---

## 피벗(Pivot)

https://en.wikipedia.org/wiki/Pivot_table

정의: 피벗 테이블은 하나 이상의 개별 카테고리 내에서 보다 광범위한 테이블(예: 데이터베이스, 스프레드시트 또는 비즈니스 인텔리전스 프로그램)의 개별 그룹을 집계한 값의 테이블 → 주로 자료의 가독성을 향상시키기 위한 작업

## 피벗을 사용하는 이유

- 반복되는 행을 별개의 열로 변환.
- 추세를 분석하고, 값을 비교하고, 데이터를 요약하기가 용이.
- Excel과 유사한 피벗 테이블을 만드는 데 활용.

피벗은 평소 자주 사용하던 메소드는 아니었지만, 데이터 프레임을 다루는 과정에서 피벗을 쓰지만 않았을 뿐 동일한 전처리 과정을 많이 수행했었음. 특정 열(특히 시간)을 인덱스로 설정한 후 원하는 feature의 추세를 확인하고 시계열 패턴을 분석하는 데 자주 사용했던 방식과 동일



<aside>
피벗과 병합은 데이터의 가독성을 향상시키고 분석가로 하여금 원하는 정보를 얻기 위해 활용할 수 있는 데이터 프레임(데이터 구조)의 형태를 변환하는 방법이다.
</aside>
