---
title : \[ML with Python\] 2장. 지도 학습 - 3절
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
  overlay_image : /assets/images/category/ml.jpg
published : false
---

2.3 지도 학습 알고리즘

본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

## 2.3 지도 학습 알고리즘

이 절에서는 가장 많이 사용하는 머신러닝 알고리즘들을 간략하게 봅니다.
알고리즘들에서 봐야할 포인트는 다음과 같습니다.

- 학습과 예측의 방법
- 모델 복잡도의 역할
- 장단점
- 적합한 데이터
- 매개변수와 옵션의 의미

지도 학습 알고리즘에 어떤 종류가 있는지를 우선으로 보면 좋을 것 같습니다.

### 2.3.1 예제에 사용할 데이터셋

앞으로 나올 알고리즘을 위해 사용되는 데이터셋은 다음과 같습니다.
일단 기본적인 인위적 데이터셋 종류만 소개하고, 실제 데이터셋은 필요할 때마다 소개하겠습니다.

#### 이진 분류 데이터셋

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

#### wave 데이터셋

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



### 2.3.2 k-최근접 이웃

**k-NN(k-Nearest Neighbors)** 알고리즘은 가장 간단한 머신러닝 알고리즘입니다.
직관적인 분류 방법으로 생각하면 됩니다. 데이터와 가장 가까운 데이터를 바탕으로 분류를 진행하는 것입니다.
훈련 데이터셋을 그냥 저장하는 것이 모델을 만드는 과정의 전부입니다.
새로운 데이터 포인트에 대해 예측할 땐 알고리즘이 훈련 데이터 셋에서 가장 가까운 데이터 포인트, 즉 `최근접 이웃`을 찾습니다.

#### k-최근접 이웃 분류

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

#### KNeighborsClassifier 분석

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

#### k-최근접 이웃 회귀

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

$$R^2 = 1 - \frac{\Sigma(y-\hat{y})^2}{\Sigma(y-\bar{y})^2}$$

#### KNeighborsRegressor 분석

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

#### 장단점과 매개변수

일반적으로 k-NN에서 중요한 매개변수는 두 개입니다. 데이터 포인트 사이의 거리를 재는 방법과 이웃의 수입니다.
거리 측정 방식은 보통 유클리디안 거리입니다. 하지만 이것도 스케일을 맞춰주는 것이 보통입니다.
이웃의 수 경우에는 3, 5 정도 적은 수에서 가장 효율이 좋지만 이도 여러번 테스트를 통해 맞춰주어야 합니다.

k-NN의 장점은 매우 쉬운 모델이라는 점입니다. 조정하지 않아도 어느 정도 성능을 발휘합니다.
복잡한 알고리즘을 하기 전 가볍게 시도할 수 있는 모델입니다. 비교적 빠르지만 훈련 세트가 매우 크면 예측이 느려집니다.
특성에 따른 스케일 전처리가 매우 중요합니다. 너무 많은 특성이 있는 데이터에서는 성능이 떨어지며 특성 값이 대부분이 0인 희소 데이터셋에서는 특히 잘 작동합니다.

예측이 느리고 많은 특성을 처리하는 능력이 부족해 현업에서는 잘 쓰지 않습니다.
이런 단점이 없는 알고리즘이 다음 모델인 **선형 모델** 입니다.

### 2.3.3 선형 모델

**선형 모델(linear model)** 은 100여 년 전에 개발되었고, 몇십 년 동안 폭넓게 연구되고 현재도 널리 쓰입니다.
선형 모델은 입력 특성에 대한 **선형 함수** 를 만들어 에측을 수행합니다.

#### 회귀와 선형 모델

회귀의 경우 선형 모델을 위한 일반화된 예측함수는 다음과 같습니다.

```
```

### 2.3.4 나이브 베이즈 분류기

### 2.3.5 결정 트리

### 2.3.6 결정 트리의 앙상블

### 2.3.7 커널 서포트 벡터 머신

### 2.3.8 신경망(딥러닝)
