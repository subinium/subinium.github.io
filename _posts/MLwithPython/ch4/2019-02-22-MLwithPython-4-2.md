---
title : \[ML with Python\] 4장 데이터 표현과 특성 공학 - 구간 분할, 이산화 그리고 선형 모델, 트리 모델
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

4.2 구간 분할, 이산화 그리고 선형 모델, 트리 모델

> 본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

데이터를 가장 잘 표현하는 방법은 데이터가 가진 의미뿐 아니라 어떤 모델을 사용하는지에 따라 다릅니다.
선형 모델과 트리 기반 모델은 특성의 표현 방식으로 인해 미치는 영향이 매우 다릅니다.

2장에서 사용한 wave 데이터셋을 다시 보겠습니다. 이 데이터는 입력 특성이 하나뿐입니다. 이 데이터셋을 이용해 선형 회귀 모델과 결정 트리 회귀를 비교해보겠습니다.

``` python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label="Decision Tree")

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), '--', label="Linear Regression")

plt.plot(X[:,0], y, 'o', c='k')
plt.ylabel("regression output")
plt.xlabel("input feature")
plt.legend(loc="best")
```

![1](https://i.imgur.com/7dbLcm8.png)

선형 모델은 선형 관계로만 모델링하므로 특성이 하나일 땐 직선으로 나타납니다. 결정 트리는 이 데이터로 훨씬 복잡한 모델을 만들 수 있습니다. 그러나 이는 데이터의 표현 형태에 따라 굉장히 달라집니다. 연속형 데이터에 아주 강력한 선형 모델을 만드는 방법 하나는 한 특성을 여러 특성으로 나누는 **구간 분할(bining)** (또는 이산화) 입니다.

이 특성의 입력값 범위가 나뉘어 여러 구간으로, 예를 들면 10개로 되어있다고 생각해봅시다. 위의 그래프에서는 -3부터 3까지이므로 구간당 간격을 0.6으로 11개의 지점을 생성하여 10개 구간을 만듭니다. 그렇다면 데이터 포인트는 일정 구간안에 속하게 될 것입니다.
우선 `np.linespace` 함수로 11개의 지점을 생성해 10개 구간을 만듭니다.

``` python
bins = np.linspace(-3, 3, 11)
print("구간 : {}".format(bins))
# 구간 : [-3.  -2.4 -1.8 -1.2 -0.6  0.   0.6  1.2  1.8  2.4  3. ]
```

그 다음 각 데이터 포인트가 어느 구간에 속하는지 기록합니다. `np.digitize` 함수를 사용하면 간단하게 계산가능합니다.
이 함수는 시작점은 포함하고, 종료점은 포함하지 않습니다.

``` python
# which_bin에는 각 데이터 포인트를 자신이 포함된 구간의 인덱스값을 가지고 있습니다.
which_bin = np.digitize(X, bins=bins)
```

위 함수를 사용하여 wave 데이터셋에 있는 연속형 특성을 각 데이터 포인트가 어느 구간에 속했는지로 인코딩한 범주형 특성으로 변환하였습니다. 이 데이터에 scikit-learn 모델을 적용하기 위해 `preprocessing` 모듈의 `OneHotEncoder`로 이산적인 이 특성을 원-핫-인코딩으로 변환합니다. `OneHotEncoder`는 `pandas.get_dummies`와 같지만 현재는 숫자로된 범주형 변수에만 적용할 수 있습니다.

``` python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
```

구간을 10개로 정했기 때문에 변환된 데이터셋 `X_binned`는 10개의 특성으로 구성됩니다.
원-핫 인코딩된 데이터로 선형 회귀 모델과 결정 트리 모델을 새로 만들어보겠습니다.

``` python
line_binned = encoder.transform(np.digitize(line, bins=bins))

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='Bined Linear Regression')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), '--', label='Bined Decision Tree')

plt.plot(X[:,0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("regression output")
plt.xlabel("input feature")
```

![2](https://i.imgur.com/0YXIwWF.png)

선형 회귀 모델과 결정 트리가 같은 예측을 만들어내서 파선과 실선이 완전히 겹쳤습니다.
구간에 대한 예측값을 나타내고 있습니다. 선형 모델의 경우에는 구간을 사용하여 유연한 그래프를 가지게 되었고, 결정 트리의 경우에는 데이터의 손실에 의해 덜 유연해졌습니다. 이와 같이 같은 변환도 모델에 따라 다르게 적용되는 것을 알 수 있습니다.

일부 특성과 출력이 비선형 관계이지만, 용향이 매우 크고 고차원 데이터셋이라 선형 모델을 사용해야 한다면 구간 분할이 모델 성능을 높이는 데 아주 좋은 방법이 될 수 있습니다.
