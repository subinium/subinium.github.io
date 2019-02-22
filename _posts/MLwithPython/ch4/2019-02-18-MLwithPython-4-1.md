---
title : \[ML with Python\] 4장 데이터 표현과 특성 공학 - 범주형 변수
category :
  - ML
tag :
  - python
  - deep-learning
  - AI
  - machine learning
  - 머신러닝
  - 데이터 전처리
  - 입문
  - subinium
  - 소스코드

sidebar:
  nav: sidebar-MLwithPython

use_math : true

header:
  teaser : /assets/images/category/ml.jpg
  overlay_color: "#AF3D8A"

---

4.1 범주형 변수

> 본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

## 4.1.0 Intro

지금까지 우리는 데이터가 2차원 실수형 배열로 각 열이 데이터 포인트를 설명하는 **연속형 특성(continuous feature)** 이라고 가정했습니다. 하지만 많은 애플리케이션에서 데이터가 이렇게 수집되지는 않습니다.

일반적인 특성의 전형적인 형태는 **범주형 특성(categorical feature)** 입니다. 또는 **이산형 특성(dicrete feature)** 이라고 하는 이 특성은 보통 숫자 값이 아닙니다. 물건을 예로 들면 컴퓨터, 옷, 책 등의 범주와 같습니다. 이런 데이터들은 연속성이 없습니다. 순서가 없고, 중간값이 없다는 것입니다.

하지만 데이터가 어떤 형태의 특성으로 구성되어 있는가보다 데이터를 어떻게 표현하는가가 머신러닝 모델의 성능에 주는 영향이 더 큽니다. 스케일의 조정도 중요하고, 특성 간 상호작용이나 일반적인 다항식을 넣는 것도 하나의 방법입니다.

특정 애플리케이션에 가장 적합한 데이터 표현을 찾는 것을 **특성 공학(feature engineeringg)** 이라 하며, 실제 문제를 해결하기 위한 주요 작업 중 하나입니다. 올바른 표현이 파라미터 조정보다 효율적인 경우가 많기에 이런 과정들은 중요합니다.

---

우선 원-핫 인코딩에서 볼 예시 데이터는 1994년 인구 조사 데이터베이스에서 추출한 미국 성인 소득 데이터셋입니다. 이 adult 데이터셋을 사용해 어떤 근로자의 수입이 50,000달라를 초과하는지, 그 이하일지를 예측하려고 합니다. 특성에는 나이, 고용형태, 교육 수준, 성별, 주당 근로시간, 직업 등이 있습니다. [다운로드는 여기로](https://github.com/rickiepark/introduction_to_ml_with_python/blob/master/data/adult.data)

이 작업은 소득이 50k 보다 큰지 작은지를 나누는 이진 분류 문제로 볼 수 있습니다. 정확한 소득을 예측한다면 회귀 문제가 되지만, 이는 어렵기에 보다 쉽게 기준을 이용해 나누는 분류로 문제를 진행합니다.

`age`와 `hour-per-week`는 연속형 특성이지만 `workclass`, `education`, `sex`, `occupation`은 범주형 데이터입니다.

맨 먼저 이 데이터에 로지스틱 회귀 분류기를 학습시켜보겠습니다. 다음 식을 이용하여 예측합니다.

$$\hat{y}=w[0] \times x[0] + w[1] \times x[1] + \cdots +w[p] \times x[p] + b > 0$$

여기서 $w[i]$와 $b$는 훈련 세트로부터 학습되는 계수이고 $x[i]$는 입력 특성입니다. 이 공식에서 $x[i]$는 숫자여야 하므로 데이터를 변환하여야 합니다.

## 4.1.1 원-핫 인코딩(가변수)

범주형 변수를 표현하는 데 가장 널리 쓰이는 방법은 **원-핫-인코딩(one-hot-encoding)** 입니다.
이를 one-out-of-N encoding 또는 가변수(dummy vatiable)라고도 합니다.
이는 범주형 변수를 0 또는 1 값을 가진 하나 이상의 새로운 특성으로 바꾼 것입니다.
0과 1로 표현된 변수는 선형 이진 분류 공식에 적용할 수 있어서, 다음과 같이 개수에 상관없이 범주마다 하나의 특성으로 표현합니다.

예를 들면 과목에 `수학`, `국어`, `과학`과 같이 3과목이 특성에 있다면 각각을 `(1, 0, 0)`, `(0, 1, 0)` `(0, 0, 1)`과 같이 표현하는 것입니다. 이런 모양 때문에 **one-hot** 또는 **one-out-of-N** 과 같이 불립니다.

머신러닝으로 훈련하기 위해서는 `수학` 과 같은 정성적인 데이터는 사용할 수 없으므로 0과 1로 바꾸는 것입니다.

> **NOTE** 원-핫 인코딩은 통계학에서 사용하는 더미 코딩과 비슷하지만 다른 점이 있습니다. 머신러닝에서는 각 범주를 각기 다른 이진 특성으로 바꾸는 반면에, 통계학에서는 k개의 값을 가진 범주형 특성을 k-1개의 특성으로 변환하는 것이 일반적입니다. (데이터 행렬의 rank 부족 현상을 피하기 위해서 입니다.)

pandas나 scikit-learn을 이용하여 범주형 변수를 원-핫 인코딩으로 바꿀 수 있습니다. 먼저 pandas를 이용해 CSV 파일에서 데이터를 읽습니다.

``` python
import os
# 이 파일은 열 이름을 나타내는 헤더가 없으므로 header=None으로 지정하고
# "names" 매개변수로 열 이름을 제공합니다
data = pd.read_csv(
    os.path.join(mglearn.datasets.DATA_PATH, "adult.data"), header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])
# 예제를 위해 몇개의 열만 선택합니다
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
             'occupation', 'income']]
# IPython.display 함수는 주피터 노트북을 위해 포맷팅된 출력을 만듭니다
display(data.head())
```
<figure>
  <img src = "https://i.imgur.com/yiMbBYW.png">
  <figcaption> IPython.display 함수는 테이블을 html 기반으로 출력해서 스샷으로 옮겨야한다. 귀찮다. </figcaption>
</figure>

### 범주형 데이터 문자열 확인하기

데이터셋을 읽고 나면, 먼저 열에 어떤 의미 있는 범주형 데이터가 있는지 확인해보는 것이 좋습니다.
외부에서 다운받은 데이터일 경우, 정해진 범주 밖에 있을 수 있고, 철자 등의 typo가 있어서 데이터를 전처리해야 할 수 있습니다.
예를 들면 전화번호에 `010-1234-5678`과 `01012345678`처럼 같은 표현이지만 다르게 표현된 데이터가 있을 수 있습니다.

열의 내용을 확인하는 가장 좋은 방법은 pandas에서 Series에 있는 `value_counts` 함수를 사용하여 유일한 값이 각각 몇 번 나타나는 지 출력해보는 것입니다.

``` python
print(data.gender.value_counts())
#  Male      21790
#  Female    10771
# Name: gender, dtype: int64
```

성별의 경우에는 Male, Female로 에러없이 잘 나누어져있고, 정확하게 두 가지 값을 지니니 원-핫-인코딩으로 나타내기 좋은 형태입니다.
실제 데이터에서는 모든 열에 대해 확인해야 합니다.

pandas에서는 `get_dummies` 함수를 사용해 데이터를 매우 쉽게 인코딩할 수 있습니다. `get_dummies` 함수는 객체 타입이나 범주형을 가진 열을 자동으로 변환해줍니다.

``` python
print("원본 특성 : \n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("get_dummies 사용 후 특성 : \n", list(data_dummies.columns))
```

```
원본 특성 :
 ['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']

get_dummies 사용 후 특성 :
 ['age', 'hours-per-week', 'workclass_ ?', 'workclass_ Federal-gov', 'workclass_ Local-gov', 'workclass_ Never-worked', 'workclass_ Private', 'workclass_ Self-emp-inc', 'workclass_ Self-emp-not-inc', 'workclass_ State-gov', 'workclass_ Without-pay', 'education_ 10th', 'education_ 11th', 'education_ 12th', 'education_ 1st-4th', 'education_ 5th-6th', 'education_ 7th-8th', 'education_ 9th', 'education_ Assoc-acdm', 'education_ Assoc-voc', 'education_ Bachelors', 'education_ Doctorate', 'education_ HS-grad', 'education_ Masters', 'education_ Preschool', 'education_ Prof-school', 'education_ Some-college', 'gender_ Female', 'gender_ Male', 'occupation_ ?', 'occupation_ Adm-clerical', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'income_ <=50K', 'income_ >50K']
```

연속형 특성인 `age`와 `hour-per-week`는 그대로지만 범주형 특성은 값마다 새로운 특성으로 확장되었습니다.

`data_dummies`의 `values` 속성을 이용해 DataFrame을 NumPy 배열로 바꿀 수 있으며, 이를 이용해 머신러닝 모델을 학습시킵니다. 모델을 학습시키기 전에 이 데이터로부터 타깃값을 분리해야 합니다. 출력값이나 출력값으로부터 유도된 변수를 특성 표현에 포함하는 것은 지도 학습 모델을 만들 때 특히 저지를기 쉬운 실수입니다.

> **CAUTION** pandas에서 열 인덱싱은 범위 끝을 포함합니다. 이와 달리 numpy는 끝 인덱싱은 범위 끝을 포함하지 않습니다.

여기서는 특성을 포함한 열, 즉 `age`부터 `occupation_ Transport-moving`까지 모든 열을 추출합니다.
이 범위에는 타깃을 뺀 모든 특성이 포함됩니다. (이 예제에서는 타깃값도 원-핫-인코딩으로 변환했지만, scikit-learn에서는 문자열도 타깃값으로 쓸 수 있으므로 imcome 열을 그대로 사용해도 됩니다.)

``` python
features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
# NumPy 배열 추출
X = features.values
y = data_dummies['income_ >50K'].values
print("X.shape: {}  y.shape: {}".format(X.shape, y.shape))
# X.shape: (32561, 44)  y.shape: (32561,)
```

이제 데이터가 scikit-learn에서 사용할 수 있는 형태가 되었으므로, 로지스틱 회귀 분류기를 이용하면 됩니다.

``` python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
print("테스트 점수: {:.2f}".format(logreg.score(X_test, y_test)))
# 테스트 점수: 0.81
```

여기서 주의할 점은 테스트 세트와 훈련 세트의 특성이 같은지 확인해야한다는 것입니다.
만약 size만 같고, DataFrame 내용이 다르다면 결과는 우리가 원하는 결과가 나오지 않을 것입니다.

## 4.1.2 숫자로 표현된 범주형 특성

adult 데이터셋에는 범주형 변수가 문자열로 인코딩되어 있습니다. 철자 오류가 날 수 있지만, 다른 한편으로는 변수가 범주형이라는 것을 확신할 수 있습니다. 하지만 저장 공간을 위하여 범주형 변수가 숫자로 된 경우가 많습니다.

예를 들면 옷은 1, 신발은 2와 같이 사용자의 선택지를 임의로 numbering한 것입니다. 이런 데이터셋의 경우에는 연속형으로 다뤄야 할지 범주형으로 다뤄야 할지 어려울 수 있습니다. 영화 관람 등급의 경우에는 범주형이지만 순서를 가지고 있고, 영화 별점의 경우에는 별 개수에 따라 연속형으로 나타낼 수도 있고, 범주형으로 나타낼 수도 있습니다.

그러므로 `get_dummies`를 사용하는 경우에는 숫자 데이터의 경우에는 `str`로 변경 후 범주형 작업을 진행할 수 있습니다.
`demo_df['숫자'] = demo_df['숫자'].astype(str)`과 같이 변경할 수 있습니다.
또는 columns 매개변수에 저장하여 만들 수도 있습니다. `pd.get_dummies(demo_df, columns=['숫자'])`
