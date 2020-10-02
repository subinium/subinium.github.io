---
title : \[ML with Python\] 2장. 지도 학습 - 지도 학습 알고리즘 (1)
category :
  - ML
tag :
  - python
  - deep-learning
  - AI
  - machine learning
  - 머신러닝
  - 지도 학습
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

2.3 지도 학습 알고리즘 (1)

> 본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

2.3절은 양이 많아 2개로 분류합니다.

## 2.3.0 INTRO

이 절에서는 가장 많이 사용하는 머신러닝 알고리즘들을 간략하게 봅니다.
알고리즘들에서 봐야할 포인트는 다음과 같습니다.

- 학습과 예측의 방법
- 모델 복잡도의 역할
- 장단점
- 적합한 데이터
- 매개변수와 옵션의 의미

지도 학습 알고리즘에 어떤 종류가 있는지를 우선으로 보면 좋을 것 같습니다.

## 2.3.1 예제에 사용할 데이터셋

앞으로 나올 알고리즘을 위해 사용되는 데이터셋은 다음과 같습니다.
일단 기본적인 인위적 데이터셋 종류만 소개하고, 실제 데이터셋은 필요할 때마다 소개하겠습니다.

### 이진 분류 데이터셋

두 개의 특성을 가진 forge 데이터셋은 인위적으로 만든 이진 분류 데이터셋입니다.
아래 코드는 데이터셋의 모든 데이터 포인트를 산점도로 그립니다. (아래 그리 참고)
가벼운 데이터셋으로 보여 일단은 로컬에서 cpu 진행할 예정입니다.

``` python
# 데이터셋을 만듭니다
X, y = mglearn.datasets.make_forge()
# 산점도를 그립니다
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["class 0", "class 1"], loc=4)
plt.xlabel("1st feature")
plt.ylabel("2nd feature")
print("X.shape: {}".format(X.shape)) # X.shape:(26, 2)
```

![1](https://i.imgur.com/7Qvaru5.png)

데이터 포인트 26개와 특성 2개를 가집니다.

### wave 데이터셋

회귀 알고리즘 설명에는 인위적으로 만든 wave 데이터셋을 사용합니다.
wave 데이터셋은 입력 특성 하나와 모델링할 타깃 변수를 가집니다.
아래 그림은 특성을 x축에 놓고 회귀의 타깃을 y축에 놓았습니다.

``` python
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("feature")
plt.ylabel("target")
# Text(0, 0.5, 'target')
```

![2](https://i.imgur.com/o539VWr.png)



## 2.3.2 k-최근접 이웃

**k-NN(k-Nearest Neighbors)** 알고리즘은 가장 간단한 머신러닝 알고리즘입니다.
직관적인 분류 방법으로 생각하면 됩니다. 데이터와 가장 가까운 데이터를 바탕으로 분류를 진행하는 것입니다.
훈련 데이터셋을 그냥 저장하는 것이 모델을 만드는 과정의 전부입니다.
새로운 데이터 포인트에 대해 예측할 땐 알고리즘이 훈련 데이터 셋에서 가장 가까운 데이터 포인트, 즉 `최근접 이웃`을 찾습니다.

### k-최근접 이웃 분류

가장 간단한 k-NN 알고리즘은 가장 가까운 훈련 데이터 포인트를 최근접 이웃으로 찾아 예측에 사용합니다.
단순히 이 훈련 데이터 포인트의 출력이 예측됩니다.
말이 어려워보이지만 단순하게 가장 가까운 포인트를 보고 클래스를 분류하는 것입니다.

``` python
mglearn.plots.plot_knn_classification(n_neighbors=1)
```

![3](https://i.imgur.com/m5upK9M.png)

여러 개의 포인트를 바탕으로 클래스를 분류할 수 있습니다. 아래는 가장 가까운 3개의 데이터 포인트를 이용한 분류입니다.

``` python
mglearn.plots.plot_knn_classification(n_neighbors=3)
```

![4](https://i.imgur.com/XJPyM45.png)

이제 `scikit-learn` 모듈을 활용하여 알고리즘 적용방법을 알아보겠습니다.
우선 일반화 성능 평가를 위하여 훈련 세트와 테스트 세트로 나눕니다.

``` python
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

다음은 `KNeighborsClassifier`을 임포트하여 객체를 만듭니다. 이웃 수 k는 3으로 설정해봅시다.

``` python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
```

이제 훈련 세트를 사용하여 분류 모델을 학습시킵니다. 여기서 학습은 이웃을 찾을 수 있도록 데이터를 저장하는 것입니다.

``` python
clf.fit(X_train, y_train)
# Output :
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#           metric_params=None, n_jobs=None, n_neighbors=3, p=2,
#           weights='uniform')
```

테스트 데이터에 대해 `predict` 메서드를 호출해서 예측합니다. 테스트 세트의 각 데이터 포인트에 대해 훈련 세트에서 가장 가까운 이웃을 계산한 다음 가장 많은 클래스를 찾습니다.

``` python
print("테스트 세트 예측: {}".format(clf.predict(X_test)))
# 테스트 세트 예측: [1 0 1 0 1 0 0]
```

모델이 얼마나 잘 일반화되었는지 평가를 위해 `score` 메서드에 테스트 데이터와 테스트 레이블을 넣어 호출합니다.

``` python
print("테스트 세트 정확도: {:.2f}".format(clf.score(X_test, y_test)))
# 테스트 세트 정확도: 0.86
```

7개 중에서 6개, 86%의 정확도를 가지는 것을 알 수 있습니다.

### KNeighborsClassifier 분석

2차원 데이터셋이므로 가능한 모든 테스트 포인트의 예측을 xy 평면에 그려볼 수 있습니다. 그리고 각 데이터 포인트가 속한 클래스에 따라 평면에 색을 칠합니다. 그렇게 **결정 경계(decision boundary)** 를 확인할 수 있습니다. 다음은 이웃의 수 값 k에 따라 생기는 결정 경계 이미지입니다.

``` python
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # fit 메소드는 self 오브젝트를 리턴합니다
    # 그래서 객체 생성과 fit 메소드를 한 줄에 쓸 수 있습니다
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
```
![5](https://i.imgur.com/5hY485k.png)

이웃의 수에 따라 결정 경계가 부드러워지는 것을 알 수 있습니다. 모델의 복잡도가 낮아지는 것입니다.
앞서 이야기한 과대적합과 과소적합, 그리고 일반화에 대한 관계를 확인해보도록 합시다.

과연 복잡한 모델, 간단한 모델의 중점을 살펴보는 것입니다.
여기서는 실제 데이터인 유방암 데이터셋을 사용합니다. 훈련 세트와 테스트 세트로 나누고, 이웃 수에 따라 평가해보겠습니다.

``` python
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# 1 에서 10 까지 n_neighbors 를 적용
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="train_acc")
plt.plot(neighbors_settings, test_accuracy, label="test_acc")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()
```
![6](https://i.imgur.com/LfKdXoV.png)

그래프에서 보면 알 수 있듯이 이웃의 수가 너무 적다면 모델이 복잡하여 과대적합되는 것을 알 수 있고, 이웃의 수가 많아지면 모델이 단순해지며 과소적합되는 것을 알 수 있습니다. 이렇게 머신러닝에서는 다양한 케이스에 대해서 체크하여 최적의 모델(그림에서는 이웃 수 6)을 찾아야합니다.

### k-최근접 이웃 회귀

k-최근접 이웃 알고리즘은 회귀 분석에서도 쓰입니다. wave 데이터셋에서 사용해봅시다.
최근접 점이 1개일때는 가까운 데이터 포인트와 같은 값을 가지게 되는 것입니다.

``` python
mglearn.plots.plot_knn_regression(n_neighbors=1)
```

![7](https://i.imgur.com/HuBQz1o.png)

여러 개의 최근접 모델에서는 이웃들의 평균을 사용하게 됩니다.

``` python
mglearn.plots.plot_knn_regression(n_neighbors=3)
```

![8](https://i.imgur.com/rzcAjzL.png)

scikit-learn에서는 `KNeighborsRegressor`에 구현되어 있습니다.

``` python
from sklearn.neighbors import KNeighborsRegressor
​
X, y = mglearn.datasets.make_wave(n_samples=40)
​
# wave 데이터셋을 훈련 세트와 테스트 세트로 나눕니다
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
​
# 이웃의 수를 3으로 하여 모델의 객체를 만듭니다
reg = KNeighborsRegressor(n_neighbors=3)
# 훈련 데이터와 타깃을 사용하여 모델을 학습시킵니다
reg.fit(X_train, y_train)

# Output :
# KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
#           metric_params=None, n_jobs=None, n_neighbors=3, p=2,
#           weights='uniform')
```

그리고 예측과 예측의 결과입니다.

``` python
print("테스트 세트 예측:\n{}".format(reg.predict(X_test)))
print("테스트 세트 R^2: {:.2f}".format(reg.score(X_test, y_test)))

# 테스트 세트 예측: [-0.05396539  0.35686046  1.13671923 -1.89415682 -1.13881398 -1.63113382 0.35686046  0.91241374 -0.44680446 -1.13881398]
# 테스트 세트 R^2: 0.83
```

여기서 R^2는 결정 계수로 회귀 모델에서 예측의 적합도를 0과 1 사이의 값으로 계산한 것입니다.
1이 예측이 완벽한 경우, 0이 훈련 세트의 출력값인 y_train의 평균으로만 예측하는 모델의 경우입니다.
자세한 식은 아래와 같습니다. $y$는 타깃값, $\bar{y}$는 타깃값의 평균값, $\hat{y}$는 모델의 예측값입니다.

$$R^2 = 1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$$

### KNeighborsRegressor 분석

1차원 데이터셋에 대해 가능한 모든 특성 값을 만들어 예측해볼 수 있습니다. 이를 위해 x 축을 따라 많은 포인트를 생성해 테스트 데이터셋을 만듭니다.

``` python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# -3 과 3 사이에 1,000 개의 데이터 포인트를 만듭니다
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 1, 3, 9 이웃을 사용한 예측을 합니다
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title(
        "{} neighbors score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
    ax.set_xlabel("feature")
    ax.set_ylabel("target")
axes[0].legend(["model_predict", "train_data/target", "test_data / target"], loc="best")
```

![9](https://i.imgur.com/oquUQMk.png)

이웃을 많이 사용할 수록 훈련 데이터에는 안맞을 수 있지만, 더 안정된 예측을 얻는 것을 알 수 있습니다.

### 장단점과 매개변수

일반적으로 k-NN에서 중요한 매개변수는 두 개입니다. 데이터 포인트 사이의 거리를 재는 방법과 이웃의 수입니다.
거리 측정 방식은 보통 유클리디안 거리입니다. 하지만 이것도 스케일을 맞춰주는 것이 보통입니다.
이웃의 수 경우에는 3, 5 정도 적은 수에서 가장 효율이 좋지만 이도 여러번 테스트를 통해 맞춰주어야 합니다.

k-NN의 장점은 매우 쉬운 모델이라는 점입니다. 조정하지 않아도 어느 정도 성능을 발휘합니다.
복잡한 알고리즘을 하기 전 가볍게 시도할 수 있는 모델입니다. 비교적 빠르지만 훈련 세트가 매우 크면 예측이 느려집니다.
특성에 따른 스케일 전처리가 매우 중요합니다. 너무 많은 특성이 있는 데이터에서는 성능이 떨어지며 특성 값이 대부분이 0인 희소 데이터셋에서는 특히 잘 작동합니다.

예측이 느리고 많은 특성을 처리하는 능력이 부족해 현업에서는 잘 쓰지 않습니다.
이런 단점이 없는 알고리즘이 다음 모델인 **선형 모델** 입니다.

## 2.3.3 선형 모델

**선형 모델(linear model)** 은 100여 년 전에 개발되었고, 몇십 년 동안 폭넓게 연구되고 현재도 널리 쓰입니다.
선형 모델은 입력 특성에 대한 **선형 함수** 를 만들어 에측을 수행합니다.

### 회귀와 선형 모델

회귀의 경우 선형 모델을 위한 일반화된 예측함수는 다음과 같습니다.

$$\hat{y} = w[0]\times x[0] + w[0]\times x[0] + \cdots + w[p]\times x[p] + b$$

이 식에서 $x[i]$는 하나의 데이터 포인트에 대한 특성을 의미하고, $w[i]$는 그에 따른 가중치 그리고 $b$는 bias(편향)를 의미합니다. 즉 $w, b$값을 학습하는 것이 목표입니다. 그리고 $\hat{y}$은 예측값을 의미합니다.

특성이 하나인 데이터셋이라면 이 식은 다음과 같습니다.

$$\hat{y} = w[0]\times x[0] + b$$

위 식은 $w[0]$ 인 기울기와 $b$ 절편을 가지는 직선 방정식을 의미합니다.
wave 데이터셋에서 확인해보겠습니다.

``` python
mglearn.plots.plot_linear_regression_wave()
# w[0]: 0.393906  b: -0.031804
```

![10](https://i.imgur.com/VeMSzrj.png)

회귀를 위한 선형 모델은 특성이 하나일 땐 직선, 두 개일 땐 평면이 되며, 더 높은 차원에서는 초평면(hyperplane)이 되는 회귀모델의 특징을 가지고 있습니다.

위에 있는 KNeighborsRegressor를 사용한 선보다는 선형 모델의 직선이 더 제약이 많아보입니다. 즉 데이터가 가지고 있던 상세 정보를 잃어버린 것처럼 보입니다. 1차원 데이터만 봐서는 그럴 수 있지만 특성이 많아질수록 선형 함수는 모델링을 더 구체적이고 확실하게 할 수 있습니다.

회귀를 위한 선형 모델은 다양합니다. 이 모델들은 훈련 데이터로부터 w와 b를 학습하는 방법과 모델의 복잡도를 제어하는 방법에서 차이가 납니다. 이제 회귀에서 사용되는 다양한 선형 모델을 봅시다.

### 선형 회귀(최소제곱법)

**선형 회귀(linear regression)** 또는 **최소제곱법(OLS, ordinary least squares)** 은 가장 간단하고 오래된 회귀용 선형 알고리즘입니다. 선형 회귀는 예측과 훈련 세트에 있는 타깃 y 사이의 **평균제곱오차(MSE, mean squared error)** 를 최소화하는 파라미터 w, b를 찾습니다.

그렇다면 평균제곱오차는 무엇인가? 이는 예측값과 차이값의 차이를 제곱하여 더한 후에 샘플의 개수로 나눈 것입니다. 식으로 적으면 다음과 같습니다.

$$MSE = \frac{1}{n}\sum_{1}^{n}(y_i - \hat{y}_i)^2 $$

선형 회귀는 매개변수가 없는 것이 장점이지만, 그렇기에 모델의 복잡도를 제어할 방법도 없습니다.
다음은 scikit-learn을 이용하여 선형모델을 만드는 코드입니다.

``` python
from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
​
lr = LinearRegression().fit(X_train, y_train)
```

기울기 파라미터(w)는 **가중치(weight)** 또는 **계수(coefficient)** 라고 하며 `lr` 객체의 `coef_` 속성에 저장되어 있고 편향(b) `intercept_` 속성에 저장되어 있습니다.

> **NOTE** scikit-learn은 훈련 데이터에서 유도된 속성은 항상 끝에 밑줄( _ ) 을 붙입니다. 그 이유는 사용자가 지정한 매개변수와 구분하기 위해서입니다.

``` python
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
# lr.coef_: [0.39390555]
# lr.intercept_: -0.031804343026759746
```

여기서 `intercept_` 속성은 실수 값 하나이지만, `coef_` 속성은 각 입력 특성에 하나씩 대응되는 NumPy 배열입니다.

훈련 세트와 테스트 세트의 성능을 확인해보겠습니다.

``` python
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
# 훈련 세트 점수: 0.67
# 테스트 세트 점수: 0.66
```

R^2 값이 그렇게 좋은 값은 아닙니다. 하지만 훈련 세트와 테스트 세트의 점수가 매우 비슷한 것을 알 수 있습니다. 이는 과대적합이 아닌 과소적합인 상태를 의미합니다. 1차원 데이터셋에서는 모델이 매우 단순하므로 과대적합을 걱정할 필요가 없습니다. 하지만 고차원 데이터셋에서는 선형 모델의 성능이 매우 높아져서 과대적합될 가능성이 높습니다.

LinearRegression 모델이 복잡한 데이터셋에서는 어떻게 되는지 확인하기 위해 이번에는 보스턴 주택가격 데이터셋을 사용합니다. 샘플이 506개가 있고 특성은 유도된 것을 합쳐 104개가 있습니다.
데이터셋을 읽고 훈련 세트와 테스트 세트로 나눕니다.

``` python
X, y = mglearn.datasets.load_extended_boston()
​
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
```

여기서도 성능을 확인해봅시다.

``` python
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
# 훈련 세트 점수: 0.95
# 테스트 세트 점수: 0.61
```

여기서는 훈련 세트에서는 점수가 높지만 테스트에는 점수가 낮은 것을 확인할 수 있습니다.
이는 과대적합의 확실한 신호이므로 모델의 복잡도를 제어할 수 있는 모델을 사용해야합니다.
기본 선형 회귀 방식 대신 가장 널리 쓰이는 모델은 이제 볼 **릿지** 회귀입니다.

### 릿지 회귀(능형 회귀)

**릿지(Ridge)** 도 회귀를 위한 선형 모델이므로 최소적합법에서 사용한 것과 같은 예측 함수를 사용합니다. 하지만 릿지 회귀에서의 가중치(w) 선택은 훈련 데이터를 잘 예측하기 위해서 뿐만 아니라 추가 제약 조건을 만족시키기 위한 목적도 있습니다. 가중치의 절대값을 가능한 한 작게 만드는 것입니다.

직관적으로 생각하면 모든 특성이 출력에 주는 영향을 최소한으로 만듭니다.
이런 제약을 **규제(regularization)** 라고 합니다. 규제란 과대적합이 되지 않도록 모델을 강제로 제한한다는 의미입니다. 릿지에서는 L2 규제를 사용합니다. (수학적으로 릿지는 계수의 L2 norm의 제곱을 패널티로 적용합니다.)

정확히는 평균제곱오차 식에 $\alpha \sum^m_{j=1}w_j^2$ 항이 추가됩니다. $\alpha$의 크기가 커지면 페널티의 효과가 커지고, 작아지면 효과가 작아집니다.

릿지 회귀는 `linear_model.Ridge`에 구현되어 있습니다. 릿지 회귀로 확장된 보스턴 주택가격 데이터셋에 적용해봅시다.

``` python
from sklearn.linear_model import Ridge
​
ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))
# 훈련 세트 점수: 0.89
# 테스트 세트 점수: 0.75
```

훈련 세트에서는 LinearRegression보다 점수가 낮지만, 테스트 세트에 대한 점수는 더 높습니다. 덜 자유로운 모델인만큼 과대적합이 적어집니다. 모델의 복잡도가 낮아지면 훈련 세트에 대한 성능은 나빠지지만 더 일반화된 모델이 됩니다.

이제는 alpha값을 바꿔 더 규제를 진행해봅시다. 위에서 사용한 코드에서는 alpha의 default값으로 1.0으로 설정되었습니다.

``` python
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge10.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge10.score(X_test, y_test)))
# 훈련 세트 점수: 0.79
# 테스트 세트 점수: 0.64
```

``` python
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge01.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge01.score(X_test, y_test)))
# 훈련 세트 점수: 0.93
# 테스트 세트 점수: 0.77
```

alpha값을 0에 가까이하면 자연스럽게 가중치 값은 0으로 근사하며 LinearRegression의 결과와 거의 같아집니다. 이 데이터에서는 0.1로 설정할 때 테스트에서 괜찮은 성능을 냈습니다.

alpha 값에 따라 `coef_` 속성이 달라지는 것을 그래프로 확인해봅시다. 목적에 부합하는 방법이 맞다면 alpha가 커질수록 `coef_` 값이 0에 가까워질 것입니다.

``` python
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
​
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("coef list")
plt.ylabel("coef size")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()
```

![11](https://i.imgur.com/2t17BLi.png)

이 그래프는 x축에 `coef_` 원소를 위치대로 나열한 것입니다. x=0에는 첫 번째 계수, x=1에는 두 번째 계수입니다. 이런 식으로 x=100까지 계속됩니다. y 축은 각 계수의 수치를 나타냅니다.

alpha가 10일때는 계수가 -3과 3 사이에 위치합니다. 그리고 LinearRegression과 근사할 수록 그 계수 크기가 커지는 것을 확인할 수 있습니다.

규제 효과를 이해하는 또 다른 방법은 alpha 값을 고정하고 훈련 데이터의 크기를 변화시켜 보는 것입니다. 아래 코드는 보스턴 주택가격 데이터셋에서 여러 가지 크기로 샘플링하여 LinearRegression과 Ridge(alpha=0)을 적용한 것입니다. 이렇게 데이터셋의 크기에 따른 모델의 성능 변화를 나타낸 그래프를 **학습 곡선(learning curve)** 라고 합니다.

> **NOTE** 훈련 과정을 여러 번 반복하면서 학습하는 알고리즘에서는 반복의 횟수에 따른 성능 변화를 나타내는 그래프를 학습 곡선이라고 합니다.

``` python
mglearn.plots.plot_ridge_n_samples()
```

![12](https://i.imgur.com/GmMihKJ.png)

(예상대로) 모든 데이터셋에 대해 릿지와 선형 회귀 모두 훈련 세트의 점수가 테스트 세트의 점수보다 높습니다.
릿지에는 규제가 적용되므로 릿지의 훈련 데이터 점수가 전체적으로 선형 회귀의 훈련 데이터 점수보다 낮습니다.

그러나 테스트 데이터셋에서는 릿지가 점수가 더 높으며 작은 데이터셋에서는 더 그렇습니다.
데이터셋 크기가 400 미만에서는 선형회귀가 어떤 것도 학습하고 있지 못합니다.
데이터가 많아짐에 따라 규제 항은 덜 중요해져서 릿지 회귀와 선형 회귀의 성능이 같아진다는 점을 알 수 있습니다.
여기서 흥미로운 점은 선형 회귀의 훈련 데이터 성능이 감소한다는 것입니다. 이는 데이터가 많아질수록 모델이 데이터를 기억하거나 과대적합하기 어려워지기 때문입니다.

### 라쏘

규제를 적용하는 또 다른 방법은 **Lasso** 입니다. 릿지 회귀에서와 같이 계수를 0에 가깝게 만들려고 합니다. 하지만 방식이 조금 다르며 이를 L1 규제라고 합니다. (계수의 절댓값 합입니다.)

정확히는 평균제곱오차 식에 $\alpha \sum^m_{j=1}|w_j|$ 항이 추가됩니다. 릿지와 마찬가지로 $\alpha$의 크기가 커지면 페널티의 효과가 커지고, 작아지면 효과가 작아집니다.

여기서는 실제로 계수가 0이 됩니다. 완전히 제외되는 특성이 생긴다는 뜻입니다. 어떻게 보면 특성 선택(feature selection)이 자동으로 이뤄진다고 볼 수 있습니다. 일부 계수를 0으로 만들면 모델을 이해하기 쉬워지고 이 모델의 가장 중요한 특성이 무엇인지 드러내줍니다.

확장된 보스턴 주택가격 데이터셋에 라쏘를 적용해보겠습니다.

``` python
from sklearn.linear_model import Lasso
​
lasso = Lasso().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso.score(X_test, y_test)))
print("사용한 특성의 개수: {}".format(np.sum(lasso.coef_ != 0)))
# 훈련 세트 점수: 0.29
# 테스트 세트 점수: 0.21
# 사용한 특성의 개수: 4
```

훈련 세트 그리고 테스트 세트 모두 점수가 별로 좋지 않습니다. 과소적합이며 104개의 특성 중 4개만 사용한 것을 볼 수 있습니다. Ridge와 마찬가지로 Lasso도 계수를 얼마나 강하게 0으로 보낼지를 조절하는 alpha 매개변수를 지원합니다. 여기서도 default 값은 1.0입니다. 과소적합을 줄이기 위해 alpha값을 줄여가며 체크합니다. 이렇게 하려면 `max_iter`의 기본값을 늘려야 합니다.

> **NOTE** Lasso는 L1, L2 규제를 함께 쓰는 엘라스틱넷 방식에서 L2 규제가 빠진 것입니다. Lasso의 alpha 매개변수는 R의 엘라스틱넷 패키지인 glmnet의 lambda 매개변수와 같은 의미입니다. 경사하강법을 바탕으로 최적값을 찾아나가므로 alpha를 줄이면 가장 낮은 오차를 찾아가는 반복 횟수가 늘어가게 됩니다. 이는 Lasso 객체의  `n_iter_` 속성에 저장되어 있습니다.

``` python
# "max_iter" 기본 값을 증가시키지 않으면 max_iter 값을 늘이라는 경고가 발생합니다
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso001.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso001.score(X_test, y_test)))
print("사용한 특성의 개수: {}".format(np.sum(lasso001.coef_ != 0)))
# 훈련 세트 점수: 0.90
# 테스트 세트 점수: 0.77
# 사용한 특성의 개수: 33
```

Ridge보다는 성능이 좋은 것을 확인할 수 있습니다. 104개중 33개의 특성만 사용하니 모델을 분석하기 비교적 쉬워집니다. 하지만 alpha를 너무 낮추면 규제의 효과가 없어져 과대적합이 되므로 LinearRegression의 결과와 비슷해집니다.

``` python
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("사용한 특성의 개수: {}".format(np.sum(lasso00001.coef_ != 0)))
# 훈련 세트 점수: 0.95
# 테스트 세트 점수: 0.64
# 사용한 특성의 개수: 96
```

104개 중 96개의 특성을 사용하니 사실상 LinearRegression과 크게 다르지 않습니다. 과대적합이 일어나는 것도 확인할 수 있습니다. 위 결과들을 바탕으로 그래프를 그려보겠습니다.

![13](https://i.imgur.com/wCQdYvs.png)

그래프에서 보면 알 수 있듯이 Lasso에서 alpha값이 1.0, 0.01 때는 많은 특성이 0임을 알 수 있습니다.
또한 alpha가 매우 작은 값(0.0001)이 되면 계수 대부분이 규제받지 않은 모델이 됩니다.
Ridge 0.1과 Lasso 0.01 모델은 비슷한 성능을 내긴했지만 Ridge의 계수 중 0인 계수는 없습니다.

Ridge와 Lasso 모델 중에서는 Ridge를 더 선호합니다. 하지만 특성이 많고 일부분만 중요하다면 Lasso가 더 좋은 선택일 수 있습니다. 쉽게 해석 가능한 모델을 만들 수 있는 것입니다.

이 두 모델의 패널티를 결합하여 최상의 성능을 내는 **엘라스틱넷(ElasticNet)** 모델도 scikit-learn에서 제공합니다. 하지만 L1과 L2에 대한 매개변수를 각각 조절해야합니다.

### 분류용 선형 모델

선형 모델은 분류에도 널리 사용합니다. 분류 중 가장 기본인 이진 분류(binary classification)을 봅시다. 이 경우 예측을 위한 방정식은 다음과 같습니다.

$$\hat{y}=w[0] \times x[0] + w[1] \times x[1] + \cdots +w[p] \times x[p] + b > 0$$

선형 회귀와 크게 다른 점은 없습니다. 부등식이 추가되었다는 정도가 있습니다.
분류 문제에서는 함수에서 계산한 값이 0보다 크면 +1, 작으면 -1로 예측합니다.
이 규칙은 분류에 쓰이는 모든 선형모델에서 동일합니다. (시그모이드 함수의 경우에는 조금 다르지만 거의 유사합니다.) 여기에서도 계수(w)와 편향(절편, b)을 찾기 위한 방법이 많이 있습니다.

회귀용 선형 모델에서는 출력 $\hat{y}$이 특성의 선형 함수였습니다. 즉 직선, 평면, 초평면입니다. 분류용 선형 모델에서는 **결정 경계** 가 입력의 선형 함수입니다. 직선, 평면, 초평면이 클래스를 분류해주는 경계라는 의미입니다.
이런 선형 모델을 학습시키는 알고리즘은 다양한데, 다음은 두 방법으로 구분할 수 있습니다.

- 특정 계수와 절편의 조합이 훈련 데이터에 얼마나 잘 맞는지 측정하는 방법
- 사용할 수 있는 규제가 있는지, 있다면 어떤 방식인지

가장 널리 알려진 두 개의 선형 분류의 알고리즘은 `linear_model.LogisticRegression`에 구현된 **로지스틱 회귀(logistic regression)** 와 `svm.LinearSVC`에 구현된 선형 **서포트 벡터 머신(support vector machine)** 입니다. 로지스틱 ***회귀*** 는 회귀가 아닌 분류입니다.

forge 데이터셋을 사용하여 위 두 모델을 만들고 결정 경계를 그림으로 나타내봅시다.

``` python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend()
```

![14](https://i.imgur.com/WXi5TOP.png)

그림에서 직선은 클래스를 나누는 결정 경계입니다. 여기서는 위가 1, 아래가 0 클래스임을 알 수 있습니다.
분류 자체는 비슷하게 해낸 것을 알 수 있습니다. 두 모델 모두 2개의 잘못된 분류가 있습니다. 둘 모두 L2 규제를 기본으로 사용하고 있습니다.

이 모델들에서 규제의 강도를 결정하는 매개변수는 `C`입니다. `C`값이 높아지면 규제가 감소합니다. 값이 작아지면 데이터 포인트 중 다수에 맞추려고 하며, 커지면 개개의 데이터 포인트를 정확히 분류하려고 합니다.
`mglearn`을 이용하여 LinearSVC 예시를 확인해봅시다.

``` python
mglearn.plots.plot_linear_svc_regularization()
```

![15](https://i.imgur.com/YvbrQmV.png)

결과를 확인해보면 C의 값이 큰 오른쪽 그래프에서는 훈련 데이터에 대해서 잘 분류하고 있지만, 전체적 배치 특성에 비해 과한 분류 직선을 가지고 있습니다. 과대적합임을 추측할 수 있습니다.

회귀와 비슷하게 분류에서의 선형 모델은 낮은 차원의 데이터에서는 결정 경계가 직선이나 평면이어서 매우 제한적인 것처럼 보이지만, 고차원에서는 초평면을 가지기에 매우 강력한 모델임은 분명합니다. 하지만 그만큼 과대적합의 위험이 커지는 단점도 있습니다.
유방암 데이터를 사용하여ㅛ LogisticRegression을 좀 더 자세히 분석해보겠습니다.

``` python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression( ).fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(logreg.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg.score(X_test, y_test)))
# 훈련 세트 점수: 0.955
# 테스트 세트 점수: 0.958
```

`C = 1`인 기본적인 모델에서 훈련, 테스트 세트 모두 95% 정확도로 꽤 훌륭한 성능을 내고 있습니다. 하지만 훈련 세트와 테스트 세트의 성능이 비슷하므로 과소적합인 것 같습니다. 모델의 제약을 더 풀어주기 위해 C를 100으로 증가시켜봅시다.

``` python
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(logreg100.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg100.score(X_test, y_test)))
# 훈련 세트 점수: 0.974
# 테스트 세트 점수: 0.965
```

과소적합이 어느정도 해소됨과 동시에 정확도도 올라간 것을 확인할 수 있습니다. 이 데이터셋에서는 복잡도가 높은 모델일수록 성능이 좋아지는 것을 알 수 있습니다. 그렇다면 규제를 더 많이한다면 어떻게 될까요?

``` python
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(logreg001.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg001.score(X_test, y_test)))
# 훈련 세트 점수: 0.934
# 테스트 세트 점수: 0.930
```

93% 정확도도 낮은 것은 아니지만 정확도가 떨어지는 것을 확인할 수 있습니다. 이제 규제 매개변수 C 설정을 세 가지로 다르게 하여 학습시킨 모델의 계수를 그래프로 확인해봅시다.

``` python
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel("feature")
plt.ylabel("coef size")
plt.legend()
```

![16](https://i.imgur.com/bca88Qq.png)

대부분 비슷한 부호의 가중치에 크기만 변화하지만 일부 음양이 달라지는 계수가 존재합니다. 그런 특성의 경우에는 병의 양성, 음성에 미치는 영향을 판단하기 어렵다는 것을 알 수 있습니다.

더 이해하기 쉬운 모델을 원한다면 L1 규제를 사용하면 됩니다. 일부 특성이 사라지겠지만 분석하기 좀 더 쉬워집니다. 다음은 그 결과입니다.

``` python
for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
    print("C={:.3f} 인 l1 로지스틱 회귀의 훈련 정확도: {:.2f}".format(
          C, lr_l1.score(X_train, y_train)))
    print("C={:.3f} 인 l1 로지스틱 회귀의 테스트 정확도: {:.2f}".format(
          C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
​
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.xlabel("feature")
plt.ylabel("coef size")
​
plt.ylim(-5, 5)
plt.legend(loc=3)

# C=0.001 인 l1 로지스틱 회귀의 훈련 정확도: 0.91
# C=0.001 인 l1 로지스틱 회귀의 테스트 정확도: 0.92
# C=1.000 인 l1 로지스틱 회귀의 훈련 정확도: 0.96
# C=1.000 인 l1 로지스틱 회귀의 테스트 정확도: 0.96
# C=100.000 인 l1 로지스틱 회귀의 훈련 정확도: 0.99
# C=100.000 인 l1 로지스틱 회귀의 테스트 정확도: 0.98
```

![17](https://i.imgur.com/tG6n5sC.png)

### 다중 클래스 분류용 선형 모델

로지스틱 회귀를 제외하고 많은 선형 분류 모델은 이진 분류만을 지원합니다.

> 로지스틱 회귀의 경우에는 소프트맥스(softmax) 함수를 사용한 다중 클래스 분류 알고리즘을 지원합니다

그렇기에 다중 클래스로 확장하는 보편적인 방법은 일대다 방법입니다. N개의 클래스를 위해서는 N개의 분류기가 있으면 됩니다. 클래스의 수만큼 분류기를 작동하여 가장 높은 점수를 내는 분류기의 클래스를 예측값으로 선택하는 것입니다.

클래스별 이진 분류기를 만들면 각 클래스가 계수 벡터와 절편을 하나씩 갖게 됩니다. 결국 분류 신뢰도를 나타내는 다음 공식의 값이 가장 높은 클래스가 해당 데이터의 클래스 레이블로 할당됩니다.

$$w[0] \times x[0] + w[1] \times x[1] + \cdots +w[p] \times x[p] + b$$

다중 클래스 로지스틱 회귀에서는 일대다 방식과는 조금 다릅니다. 하지만 여기서도 클래스마다 하나의 계수 벡터와 절편을 만들며, 예측 방법도 같습니다.

참고로 다중 클래스 로지스틱 회귀를 위한 공식은 $Pr(Y_i=c)=\frac{e^{w_c \cdot X_i}}{\sum_{k=1}^K e^{w_k \cdot X_i}}$ 입니다. $K$는 클래스의 수, $X$는 데이터 포인트, $Y$는 출력 입니다.

세 개의 클래스를 가진 간단한 데이터셋에 일대다 방식을 적용해보겠습니다. 이 데이터셋은 2차원이며 각 클래스의 데이터는 정규분포를 따릅니다.

``` python
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(["class 0", "class 1", "class 2"])
```

![18](https://i.imgur.com/eDTn6To.png)

이 데이터셋으로 LinearSVC 분류기를 훈련해보겠습니다.
세 개의 이진 분류기가 만드는 경계를 시각화해보겠습니다.

``` python
linear_svm = LinearSVC().fit(X, y)
print("계수 배열의 크기: ", linear_svm.coef_.shape)
print("절편 배열의 크기: ", linear_svm.intercept_.shape)

# 계수 배열의 크기:  (3, 2)
# 절편 배열의 크기:  (3,)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(['class 0', 'class 1', 'class 2', '0 border', '1 border',
            '2 border'], loc=(1.01, 0.3))
```

![19](https://i.imgur.com/O9cVtvz.png)

각각의 분류기는 성공적으로 분류한 것을 확인할 수 있습니다. 하지만 중앙 삼각형 또는 겹치는 부분에 대해서는 어떻게 적용할 수 있을까요? 이 경우에는 가까운 직선의 점수가 더 높을 것이니 두 직선의 각이등분선으로 또 다시 분류가 될 것을 예측할 수 있습니다. 그렇다면 중앙 삼각형의 외심을 기준으로 나눠지는 것을 확인할 수 있을 것입니다. 이를 그려봅시다.

``` python
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(['class 0', 'class 1', 'class 2', '0 border', '1 border',
            '2 border'], loc=(1.01, 0.3))
```

![20](https://i.imgur.com/p7AVzQk.png)

### 장단점과 매개변수

선형 모델의 주요 매개변수는 회귀 모델에서는 alpha였고, 분류에서는 C입니다. alpha 값은 클수록, C 값은 작을수록 모델이 단순해집니다. 보통 두 매개변수모두 로그 스케일(1, 10, 100, 0.01, 0.001 ...)로 최적치를 정합니다.

그리고 L1 또는 L2 규제를 사용할지 정해야 합니다. 특성을 많이 사용하면 L2, 주요 특성을 줄이기 위해서는 L1을 사용합니다. L1을 사용하면 모델을 분석하기 용이해지고 중요한 특성이 무엇이며, 그 영향을 파악하기 쉽습니다.

선형 모델은 학습 속도가 빠르고 예측도 빠릅니다. 매우 큰 데이터셋과 희소한 데이터셋에도 잘 작동합니다.
선형 모델의 또 하나의 장점은 앞서 회귀와 분류에서 본 공식을 사용해 예측이 어떻게 만들어지는지 비교적 쉽게 이해할 수 있다는 것입니다. 하지만 계수의 값들에 대한 이해는 어려울 수 있습니다. 특히 데이터 특성들이 연관성이 깊을수록 그렇습니다.

샘플에 비해 특성이 많을 때 잘 작동합니다. 다른 모델로 학습하기 어려운 매우 큰 데이터셋에도 선형 모델을 많이 사용합니다. 그러나 저차원의 데이터셋에서는 다른 모델들의 일반화 성능이 더 좋습니다. 후에 SVM에서 선형 모델이 실패하는 예를 보도록 하겠습니다.

## 2.3.4 나이브 베이즈 분류기

> 책에서 매우 간단하게 언급하고 넘어갑니다. 제가 생략한게 아님을 알아주세요. :-(

**나이브 베이즈(naive bayes)** 분류기는 앞 절의 선형 모델과 매우 유사합니다. LogisticRegression이나 LinearRegression 같은 선형 분류기보다 훈련 속도가 빠른 편이지만 그 대신 일반화 성능이 조금 부족합니다.

나이브 베이즈 분류기가 효과적인 이유는 각 특성을 개별로 취급해 파라미터를 학습하고 각 특성에서 클래스별 통계를 단순하게 취합하기 때문입니다. scikit-learn에 구현된 나이브 베이즈 분류기는 3가지 입니다.

- GaussianNB : 연속적 데이터
- BernoulliNB : 이진 데이터
- MultinomialNB : 카운트 데이터

### 장단점과 매개변수

MultinomialNB와 BernoulliNB는 모델의 복잡도를 조절하는 alpha 매개변수 하나를 가지고 있습니다.
alpha가 주어지면 알고리즘이 모든 특성에 양의 값을 가지는 데이터 포인트를 alpha 개수만큼 추가합니다.
이로 통계 데이터를 완만하게 만들어주지만, 성능 변동은 비교적 크지 않아서 정확도를 어느 정도 높일때 사용합니다.
GaussianNB은 대부분 매우 고차원 데이터셋에 사용합니다.

선형 모델의 장단점과 거의 비슷합니다. 희소한 고차원 데이터에서 잘 작동하며 비교적 매개변수에 민감하지 않습니다. 선형 모델로는 학습 시간이 너무 오래 걸리는 매우 큰 데이터셋에는 나이브 베이즈 모델을 시도해볼 만하며 종종 사용됩니다.
