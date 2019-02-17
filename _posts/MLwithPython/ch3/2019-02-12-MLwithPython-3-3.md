---
title : \[ML with Python\] 3장 비지도 학습과 데이터 전처리 - 데이터 전처리와 스케일 조정
category :
  - ML
tag :
  - python
  - deep-learning
  - AI
  - machine learning
  - 머신러닝
  - 비지도 학습
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

3.3 데이터 전처리와 스케일 조정

> 본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

## 3.3.1 여러 가지 전처리 방법

우선 다음과 같은 데이터를 살펴봅시다.

![1](https://i.imgur.com/cZWXUoL.png)

다음 데이터는 인위적으로 만든 이진 분류 데이터셋입니다.
첫 번째 특징은 10과 15 사이, 두 번째 특징은 1과 9 사이에 있습니다.

오른쪽의 네 그래프는 데이터를 원하는 스케일로 변환하는 네 가지 방법을 보여줍니다.

- **StandardScaler** : 각 특성의 평균을 0, 분산을 1로 변경하여 특성의 스케일을 맞춥니다.
  - 최솟값과 최댓값의 크기를 제한하지 않습니다.
  - $\frac{x-\bar{x}}{\sigma}$
- **RobustScaler** : 평균과 분산 대신에 중간 값과 사분위 값을 사용합니다.
  - 중간 값은 정렬시 중간에 있는 값을 의미하고, 사분위값은 1/4, 3/4에 위치한 값을 의미합니다.
  - 전체 데이터와 아주 동떨어진 데이터 포인트(이상치)에 영향을 받지 않습니다.
  - $\frac{x-q_2}{q_3-q_1}$ 각각이 사분위값과 중간 값입니다.
- **MinMaxScaler** : 모든 특성이 0과 1 사이에 위치하도록 데이터를 변경합니다.
  - $\frac{x-x_{min}}{x_{max}-x_{min}}$ 를 이용하여 변경합니다.
- **Normalizer** : 위와 다른 스케일 조정법으로 특성 벡터의 유클리디안 길이가 1이되도록 조정합니다.
  - 즉 길이가 1인 원 또는 구로 투영하는 것이고, 각도만이 중요할 때 적용합니다.
  - l1, l2, max 옵션을 제공하며 유클리디안 커리인 l2가 기본값입니다.

## 3.3.2 데이터 변환 적용하기

전에 언급했듯이 스케일 조정 전처리는 지도 학습 알고리즘 적용 전에 사용합니다.
scikit-learn을 사용해 SVC에서 적용해봅시다. cancer 데이터셋에 `MinMaxScaler`를 사용해봅시다.
우선 데이터를 가져와 나누어보겠습니다.

``` python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
```

이 데이터셋에는 569개의 데이터 포인트에 있고 각 데이터 포인트는 30개의 측정값으로 이뤄져있습니다.
이 데이터셋에서 샘플 425개를 훈련, 143개를 테스트 세트로 나눴습니다.

`MinMaxScaler`는 이미 파이썬 구현 클래스가 있으니 import하여 객체를 생성하면 됩니다.

``` python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

그런 다음 `fit` 메서드에 훈련 데이터를 적용합니다.
`fit`으로 scaler를 학습한다고 생각하면 좋습니다.

``` python
scaler.fit(X_train)
```

이제 이렇게 학습한 변환을 적용하려면 `transform` 메서드를 이용해 적용하면 됩니다.
새로운 표현을 위해 변환해봅시다.

``` python
# 데이터 변환
X_train_scaled = scaler.transform(X_train)
# 스케일이 조정된 후 데이터셋의 속성을 출력합니다
print("변환된 후 크기: {}".format(X_train_scaled.shape))
print("스케일 조정 전 특성별 최소값:\n {}".format(X_train.min(axis=0)))
print("스케일 조정 전 특성별 최대값:\n {}".format(X_train.max(axis=0)))
print("스케일 조정 후 특성별 최소값:\n {}".format(X_train_scaled.min(axis=0)))
print("스케일 조정 후 특성별 최대값:\n {}".format(X_train_scaled.max(axis=0)))

# 테스트 데이터 변환
X_test_scaled = scaler.transform(X_test)
# 스케일이 조정된 후 테스트 데이터의 속성을 출력합니다
print("스케일 조정 후 특성별 최소값:\n{}".format(X_test_scaled.min(axis=0)))
print("스케일 조정 후 특성별 최대값:\n{}".format(X_test_scaled.max(axis=0)))
```

변환된 결과를 우선 봅시다.

```
변환된 후 크기: (426, 30)
스케일 조정 전 특성별 최소값:
 [6.981e+00 9.710e+00 4.379e+01 1.435e+02 5.263e-02 1.938e-02 0.000e+00
 0.000e+00 1.060e-01 5.024e-02 1.153e-01 3.602e-01 7.570e-01 6.802e+00
 1.713e-03 2.252e-03 0.000e+00 0.000e+00 9.539e-03 8.948e-04 7.930e+00
 1.202e+01 5.041e+01 1.852e+02 7.117e-02 2.729e-02 0.000e+00 0.000e+00
 1.566e-01 5.521e-02]
스케일 조정 전 특성별 최대값:
 [2.811e+01 3.928e+01 1.885e+02 2.501e+03 1.634e-01 2.867e-01 4.268e-01
 2.012e-01 3.040e-01 9.575e-02 2.873e+00 4.885e+00 2.198e+01 5.422e+02
 3.113e-02 1.354e-01 3.960e-01 5.279e-02 6.146e-02 2.984e-02 3.604e+01
 4.954e+01 2.512e+02 4.254e+03 2.226e-01 9.379e-01 1.170e+00 2.910e-01
 5.774e-01 1.486e-01]
스케일 조정 후 특성별 최소값:
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0.]
스케일 조정 후 특성별 최대값:
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1.]
스케일 조정 후 특성별 최소값:
[ 0.0336031   0.0226581   0.03144219  0.01141039  0.14128374  0.04406704
  0.          0.          0.1540404  -0.00615249 -0.00137796  0.00594501
  0.00430665  0.00079567  0.03919502  0.0112206   0.          0.
 -0.03191387  0.00664013  0.02660975  0.05810235  0.02031974  0.00943767
  0.1094235   0.02637792  0.          0.         -0.00023764 -0.00182032]
스케일 조정 후 특성별 최대값:
[0.9578778  0.81501522 0.95577362 0.89353128 0.81132075 1.21958701
 0.87956888 0.9333996  0.93232323 1.0371347  0.42669616 0.49765736
 0.44117231 0.28371044 0.48703131 0.73863671 0.76717172 0.62928585
 1.33685792 0.39057253 0.89612238 0.79317697 0.84859804 0.74488793
 0.9154725  1.13188961 1.07008547 0.92371134 1.20532319 1.63068851]
```

단순히 수치만 보면 잘 안보일 수 있지만, 주의해서봐야할 점이 있습니다.
`MinMaxScaler`의 변환은 데이터를 0과 1사이로 조정하는 것이 포인트입니다.
하지만 `X_train`의 데이터는 잘 됬지만, `X_test`는 그렇지 않은 것을 확인할 수 있습니다.

이는 `scaler`가 `X_train`의 데이터를 학습하여 식 자체가 $\frac{x_{test}-x_{train\_min}}{x_{train\_max}-x_{train\_min}}$로 적용되었기 때문입니다.

스케일은 `X_train`을 기준으로 맞추어야 하므로 결론적으로는 맞는 변환 방식입니다.
만약 스케일 조정 후 특성별 최댓값이 0과 1보다 많이 벗어난다면 훈련 데이터와 테스트 데이터의 경향성이 다른 이상치가 있다는 것을 알 수 있습니다.

## 3.3.3 훈련 데이터와 테스트 데이터의 스케일을 같은 방법으로 조정하기

지도 학습 모델에서 테스트 세트를 훈련 세트와 테스트 세트에 같은 변환을 적용해야 한다는 점이 중요합니다.
다음 예에서 테스트 세트가 따로 변환을 적용할 경우 생기는 문제를 확인해보겠습니다.

``` python
from sklearn.datasets import make_blobs
# 인위적인 데이터셋 생성
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# 훈련 세트와 테스트 세트로 나눕니다
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

# 훈련 세트와 테스트 세트의 산점도를 그립니다
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1],
                c=mglearn.cm2.colors[0], label="훈련 세트", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^',
                c=mglearn.cm2.colors[1], label="테스트 세트", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("원본 데이터")

# MinMaxScaler를 사용해 스케일을 조정합니다
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일이 조정된 데이터의 산점도를 그립니다
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=mglearn.cm2.colors[0], label="훈련 세트", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^',
                c=mglearn.cm2.colors[1], label="테스트 세트", s=60)
axes[1].set_title("스케일 조정된 데이터")

# 테스트 세트의 스케일을 따로 조정합니다
# 테스트 세트의 최솟값은 0, 최댓값은 1이 됩니다
# 이는 예제를 위한 것으로 절대로 이렇게 사용해서는 안됩니다
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

# 잘못 조정된 데이터의 산점도를 그립니다
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=mglearn.cm2.colors[0], label="training set", s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1],
                marker='^', c=mglearn.cm2.colors[1], label="test set", s=60)
axes[2].set_title("잘못 조정된 데이터")

for ax in axes:
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
fig.tight_layout()
```

![2](https://i.imgur.com/sai47w4.png)

각각은 2차원 원본 데이터, 스케일 조정된 데이터, 잘못 조정된 데이터 입니다.
세번째 그래프에서 포인트들이 경향성을 무시하고 잘못 배치된 것을 확인할 수 있습니다.

> **NOTE** `fit_transform`을 사용하여 한번에 변환하는 방법도 있습니다. (아래 코드 참고)

``` python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# 메소드 체이닝(chaining)을 사용하여 fit과 transform을 연달아 호출합니다
X_scaled = scaler.fit(X_train).transform(X_train)
# 위와 동일하지만 더 효율적입니다
X_scaled_d = scaler.fit_transform(X_train)
```
## 3.3.4 지도 학습에서 데이터 전처리 효과

이제 다시 cancer 데이터셋으로 돌아가 변환된 데이터셋을 활용한 결과와 기존 결과를 비교해보겠습니다.

``` python
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=0)

svm = SVC(gamma='auto', C=100)
svm.fit(X_train, y_train)
print("테스트 세트 정확도: {:.2f}".format(svm.score(X_test, y_test)))

# 0~1 사이로 스케일 조정
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 조정된 데이터로 SVM 학습
svm.fit(X_train_scaled, y_train)

# 스케일 조정된 테스트 세트의 정확도
print("스케일 조정된 테스트 세트의 정확도: {:.2f}".format(svm.score(X_test_scaled, y_test)))

# 테스트 세트 정확도: 0.63
# 스케일 조정된 테스트 세트의 정확도: 0.97
```

아마 처음의 결과에는 SVM의 특징에 따라 과소적합되었음을 예상할 수 있지만, 같은 모델을 데이터 스케일 조정함으로 97%까지 정확도를 올렸습니다.
데이터 스케일 조정은 어려운 과정은 아니지만, 여러 스케일 조정 방법을 적용하기 위해 scikit-learn에서 제공하는 도구를 이용하는 것이 좋습니다.
모든 전처리 모델이 동일한 함수를 제공하므로 코드의 재활용이 더 쉽게 이뤄질 수 있습니다.
