---
title : \[ML with Python\] 4장 데이터 표현과 특성 공학 - 상호작용과 다항식
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

4.3 상호작용과 다항식

> 본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

특성을 추가하는 또 좋은 방식은 원본 데이터에 **상호작용(interaction)과 다항식(polynomial)** 을 추가하는 것입니다. 통계적 분석에 자주 사용하지만, 일반적인 머신러닝에서도 많이 적용합니다.

wave 데이터 셋을 통해 다시 알아보겠습니다.

저번 장에서는 선형 모델은 wave 데이터셋의 각 구간에 대해 상숫값을 학습했습니다.
그런데 이런 절편뿐만 아니라 기울기도 학습할 수 있습니다.
바로 구간으로 분할된 데이터에 원래 특성(x축)을 다시 추가하는 것입니다. 그렇다면 이 데이터는 11차원 데이터셋이 됩니다.

``` python
X_combined = np.hstack([X, X_binned])
print(X_combined.shape)
# (100, 11)

reg = LinearRegression().fit(X_combined, y)
line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label = "Linear Regression with original data added")
plt.legend(loc='best')
plt.ylabel("regression output")
plt.xlabel("input feature")
plt.plot(X[:,0],y, 'o', c='k' )
```

![3](https://i.imgur.com/CI7Canz.png)

이 예에서 각 구간의 절편과 기울기를 학습했습니다.
학습된 기울기는 음수이고 모든 구간에 걸쳐 동일합니다. X축 특성이 하나이므로 기울기도 하나입니다.

기울기가 모든 구간에서 같으니 별로 유익해 보이지 않습니다. 우리의 목표는 구간별 기울기를 통해 더 정확한 선형 회귀 그래프를 그리는 것입니다.

이런 목표를 위해서는 데이터 포인트가 있는 구간과 x축 사이의 상호작용 특성을 추가할 수 있습니다. 이 특성이 구간 특성과 원본 특성의 곱입니다. 구간 특성과 원본 특성의 곱으로 특성을 만들어봅시다.

``` python
X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)
# (100, 20)
```

이제 이 데이터셋은 총 20개의 특성을 가집니다. 이 데이터로 선형회귀를 진행해보겠습니다.

``` python
X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)
# (100, 20)
reg = LinearRegression().fit(X_product, y)
​
line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label = "LinearRegression with feature multiplying")
​
for bin in bins:
    plt.plot([bin,bin],[-3, 3], ':', c='k', linewidth=1)

plt.plot(X[:,0], y, 'o', c='k')
plt.ylabel("regression output")
plt.xlabel("input feature")
plt.legend(loc="best")
```

![4](https://i.imgur.com/ivNCejs.png)

이제 이 그림에서 볼 수 있듯 각 구간에서 절편과 기울기가 모두 다릅니다.
구간 나누기는 연속형 특성을 확장하는 방법 중 하나입니다. 원본 특성의 다항식을 추가하는 방법도 있습니다.
특성 x가 주어지면, x**2, x**3 등의 방법을 시도해볼 수 있습니다.

이 방식이 `proecessing` 모듈의 `PolynomialFeatures`에 구현되어 있습니다.

``` python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)
```

다음 코드에서는 `degree`에 설정한 10은 10차원 까지 사용함을 의미합니다. 즉 x부터 x**10까지 사용합니다.
각 특성의 차수를 알려주는 `get_feature_names` 메서드를 사용해 특성의 의미를 알 수 있습니다.

``` python
print(poly.get_feature_names())
# ['x0', 'x0^2', 'x0^3', 'x0^4', 'x0^5', 'x0^6', 'x0^7', 'x0^8', 'x0^9', 'x0^10']
```

기존 X값에 따라 값이 기하급수적으로 커질 수 있습니다.

다항식 특성을 선형 모델과 함께 사용하면 전형적인 **다항 회귀(polynomial regression)** 모델이 됩니다.

``` python
reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label="ploynomail linear regression")
plt.plot(X[:,0], y, 'o', c='k')
plt.ylabel("regression output")
plt.xlabel("input regression")
plt.legend(loc="best")
```

![5](https://i.imgur.com/MwY1W3h.png)

다항식 특성은 1차원 데이터셋에서도 매우 부드러운 곡선을 만듭니다. 그러나 고차원 다항식은 데이터가 부족한 영역에서 너무 민감하게 반응합니다. 데이터의 말단부가 극으로 향하는 것을 볼 수 있습니다.

비교를 위해 아무런 변환도 거치지 않은 원본 데이터에 커널 SVM 모델을 학습시켜보겠습니다.

``` python
from sklearn.svm import SVR

for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X,y)
    plt.plot(line,svr.predict(line), label='SVR gamma={}'.format(gamma))

plt.plot(X[:,0], y, 'o', c='k')
plt.ylabel("regression output")
plt.xlabel("input regression")
plt.legend(loc="best")
```

![6](https://i.imgur.com/9zWrP97.png)

더 복잡한 모델인 커널 SVM을 사용해 특성 데이터를 변환하지 않고 다항 회귀와 비슷한 복잡도를 가진 예측을 만들었습니다.

상호작용과 다항식을 위한 더 현실적인 애플리케이션으로 보스턴 주택 가격 데이터셋을 이용해보겠습니다. 이미 2장에서 이 데이터셋에 다항식 특성을 적용했습니다. 이제 이 특성들이 어떻게 구성되었는지 살펴보고, 다항식 특성이 얼마나 도움이 되는지 보겠습니다.

데이터를 우선 MinMaxScaler을 사용하여 스케일을 조정하고 시작합니다.

``` python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

그 후 차수를 2로 하여 다항식 특성을 뽑습니다.

``` python
poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print(X_train.shape, X_train_poly.shape)
# (379, 13) (379, 105)
```

왜 105개인가 의문이 들 수 있습니다. 1 + 13(원본 1차항) + 13C2(2차항 without 제곱항) + 13(제곱항)으로 이해할 수 있습니다.
확인은 `get_feature_names`로 할 수 있지만 생략하겠습니다.

이제 상호작용 특성이 있는 데이터와 없는 데이터에 대해 Ridge를 사용해 성능을 비교해보겠습니다.

``` python
from sklearn.linear_model import Ridge
​
ridge = Ridge().fit(X_train_scaled, y_train)
print("상호작용 특성이 없을 때 점수 {:.3f}".format(ridge.score(X_test_scaled, y_test)))
​
ridge = Ridge().fit(X_train_poly, y_train)
print("상호작용 특성이 있을 때 점수 {:.3f}".format(ridge.score(X_test_poly, y_test)))
#상호작용 특성이 없을 때 점수 0.621
#상호작용 특성이 있을 때 점수 0.753
```

상호작용이 Ridge의 성능을 높였습니다. 그러나 랜덤 포레스트 같이 더 복잡한 모델을 사용하면 또 다른 결과를 볼 수 있습니다.

``` python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_scaled, y_train)
print("상호작용 특성이 없을 때 점수 {:.3f}".format(rf.score(X_test_scaled, y_test)))

rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_poly, y_train)
print("상호작용 특성이 있을 때 점수 {:.3f}".format(rf.score(X_test_poly, y_test)))

# 상호작용 특성이 없을 때 점수 0.795
# 상호작용 특성이 있을 때 점수 0.774
```

특성을 추가하지 않아도 랜덤 포레스트는 Ridge의 성능과 맞먹습니다. 오히려 상호작용과 다항식을 추가하면 성능이 조금 줄어듭니다.
