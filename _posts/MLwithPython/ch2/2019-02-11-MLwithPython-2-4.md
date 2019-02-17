---
title : \[ML with Python\] 2장. 지도 학습 - 분류 예측의 불확실성 추정
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

2.4 분류 예측의 불확실성 추정

> 본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

## 2.4.0 INTRO

scikit-learn에서 많이 사용하는 인터페이스 중 하나는 분류기에 예측의 불확실성을 추정할 수 있는 기능입니다.
어떤 테스트 포인트에 대해 분류기가 예측한 클래스가 무엇인지 뿐만 아니라 정확한 클래스임을 얼마나 확신하는지가 중요할 때가 많습니다. 예를 들면 보안, 의료 등의 분야가 그런 예시입니다.
(틀린 예측의 결과가 심각한 피해를 만들 가능성이 있는 경우)

scikit-learn 분류기에서 불확실성을 추정할 수 있는 함수가 두 개 있습니다. `decision_function`과 `predict_proba`입니다. 대부분의 분류 클래스는 하나 이상을 제공합니다.
인위적 2차원 데이터셋을 사용해 `GradientBoostingClassifier` 분류기의 `decision_function`과 `predict_proba` 메서드의 역할을 확인해보겠습니다.

``` python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

# 예제를 위해 클래스의 이름을 "blue" 와 "red" 로 바꿉니다
y_named = np.array(["blue", "red"])[y]

# 여러개의 배열을 한꺼번에 train_test_split 에 넣을 수 있습니다
# 훈련 세트와 테스트 세트로 나뉘는 방식은 모두 같습니다.
X_train, X_test, y_train_named, y_test_named, y_train, y_test = \
    train_test_split(X, y_named, y, random_state=0)

# 그래디언트 부스팅 모델을 만듭니다
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)
```

## 2.4.1 결정 함수

이진 분류에서 `decision` 반환값 크기는 `(n_samples, )`이며 각 샘플이 하나의 실수 값을 반환합니다.

``` python
print("X_test.shape: {}".format(X_test.shape))
print("결정 함수 결과 형태: {}".format(gbrt.decision_function(X_test).shape))
# X_test.shape: (25, 2)
# 결정 함수 결과 형태: (25,)
```

이 값은 모델이 데이터 포인트가 양성 클래인 클래스 1에 속한다고 믿는 정도입니다. 양수 값은 양성 클래스를 의미하며 음수 값은 음성 클래스를 의미합니다.

``` python
# 결정 함수 결과 중 앞부분 일부를 확인합니다
print("결정 함수:\n{}".format(gbrt.decision_function(X_test)[:6]))
# 결정 함수:
# [ 4.13592629 -1.7016989  -3.95106099 -3.62599351  4.28986668  3.66166106]
```

결정 함수의 부호만 보고 예측 결과를 알 수 있습니다. 음과 양에 따라서 예측하기 때문입니다.

``` python
print("임계치와 결정 함수 결과 비교:\n{}".format(
      gbrt.decision_function(X_test) > 0))
print("예측:\n{}".format(gbrt.predict(X_test)))
# 임계치와 결정 함수 결과 비교:
# [ True False False False  True  True False  True  True  True False  True
#   True False  True False False False  True  True  True  True  True False
#  False]
# 예측:
# ['red' 'blue' 'blue' 'blue' 'red' 'red' 'blue' 'red' 'red' 'red' 'blue'
#  'red' 'red' 'blue' 'red' 'blue' 'blue' 'blue' 'red' 'red' 'red' 'red'
#  'red' 'blue' 'blue']
```

이진 분류에서 음성 클래스는 항상 `classes_` 속성의 첫 번째, 음성 클래스는 두 번째 원소입니다.
다음과 같이 변환할 수 있습니다.

``` python
# 불리언 값을 0과 1로 변환합니다
greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
# classes_에 인덱스로 사용합니다
pred = gbrt.classes_[greater_zero]
# pred 와 gbrt.predict의 결과를 비교합니다
print("pred 는 예측 결과와 같다: {}".format(np.all(pred == gbrt.predict(X_test))))
# pred 는 예측 결과와 같다: True
```

`decision_function` 값의 범위는 데이터와 모델 파라미터에 따라 달라집니다.

``` python
decision_function = gbrt.decision_function(X_test)
print("결정 함수 최소값: {:.2f} 최대값: {:.2f}".format(
      np.min(decision_function), np.max(decision_function)))
# 결정 함수 최소값: -7.69 최대값: 4.29
```

출력 범위가 임의의 값이라서 이해하긴 어렵습니다.
2차원 평면의 모든 점에 대해 `decision_function`의 값을 색으로 표현하여 결정 경계와 함께 그래프로 나타내보겠습니다.

``` python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,
                                fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
                                            alpha=.4, cm=mglearn.ReBl)
​
for ax in axes:
    # 훈련 포인트와 테스트 포인트를 그리기
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
                             markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                             markers='o', ax=ax)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
cbar.set_alpha(1)
cbar.draw_all()
axes[0].legend(["test class 0", "test class 1", "train class 0",
                "train class 1"], ncol=4, loc=(.1, 1.1))
```

예측한 결과뿐만 아니라 분류기가 얼마나 확신하는지를 알면 추가 정보를 얻게 됩니다. 그러나 결정 함수 그래프에서 두 클래스 사이의 경게를 구분하기는 어렵습니다.(범위에 대한 이해가 어렵기 때문입니다.)

![39](https://i.imgur.com/aGK1msQ.png)

## 2.4.2 예측 확률

`predict_proba`의 출력은 각 클래스에 대한 확률이고 `decision_function`의 출력보다 이해하기 더 쉽습니다. 이진 분류에서는 항상 사이즈가 `(n_samples, 2)`

``` python
print("확률 값의 형태: {}".format(gbrt.predict_proba(X_test).shape))
확률 값의 형태: (25, 2)
```

각 행의 첫 번째 원소는 첫 번째 클래스의 예측 확률이고 두번째 원소는 두 번째 클래스의 예측 확률입니다. 확률이기 때문에 `predict_proba`의 출력은 항상 0과 1 사이의 값이며 두 클래스에 대한 확률의 합은 항상 1입니다.

``` python
# predict_proba 결과 중 앞부분 일부를 확인합니다
print("Predicted probabilities:\n{}".format(
      gbrt.predict_proba(X_test[:6])))
# Predicted probabilities:
# [[0.01573626 0.98426374]
#  [0.84575649 0.15424351]
#  [0.98112869 0.01887131]
#  [0.97406775 0.02593225]
#  [0.01352142 0.98647858]
#  [0.02504637 0.97495363]]
```

50 : 50이 나올 확률은 사실상 매우 적습니다. 복잡한 데이터셋에서는 더더욱 그렇습니다.
그렇기에 확률이 더 높은 값으로 예측값이 선택됩니다.

하지만 예측 또한 과대적합, 과소적합에 따라서 그 정도가 다릅니다.
과대적합이면 잘못된 예측에 대한 확신이 강하고, 과소적합이면 그래도 50/50에 가까운 값을 가지게 될 것입니다.
일반적으로 복잡도가 낮은 모델은 예측에 불확실성이 더 많습니다. 이런 불확실성과 모델의 정확도가 동등하면 이 모델을 **보정(calibration)** 되었다고 합니다. 즉 보정된 모델에서 70% 확신을 가진 예측은 70%의 정확도를 낼 것입니다.

앞에서 사용한 데이터 셋을 사용해 결정 경계와 클래스 1의 확률을 그려보겠습니다.


``` python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

mglearn.tools.plot_2d_separator(
    gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(
    gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')
​
for ax in axes:
    # 훈련 포인트와 테스트 포인트를 그리기
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
                             markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                             markers='o', ax=ax)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
# colorbar 를 감추지 않습니다.
cbar = plt.colorbar(scores_image, ax=axes.tolist())
cbar.set_alpha(1)
cbar.draw_all()
axes[0].legend(["test class 0", "test class 1", "train class 0",
                "train_class 1"], ncol=4, loc=(.1, 1.1))
```

![40](https://i.imgur.com/FCibSYq.png)

결정 함수보다 결정 경계가 명확하게 보이는 것을 확인할 수 있습니다.

## 2.4.3 다중 분류에서의 불확실성

다중 분류도 위에서 사용한 메서드를 이용할 수 있습니다. 클래스가 3개인 iris 데이터셋에 적용해보겠습니다.

``` python
from sklearn.datasets import load_iris
​
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=42)
​
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)
print("결정 함수의 결과 형태: {}".format(gbrt.decision_function(X_test).shape))
# plot the first few entries of the decision function
print("결정 함수 결과:\n{}".format(gbrt.decision_function(X_test)[:6, :]))

# 결정 함수의 결과 형태: (38, 3)
# 결정 함수 결과:
# [[-0.52931069  1.46560359 -0.50448467]
#  [ 1.51154215 -0.49561142 -0.50310736]
#  [-0.52379401 -0.4676268   1.51953786]
#  [-0.52931069  1.46560359 -0.50448467]
#  [-0.53107259  1.28190451  0.21510024]
#  [ 1.51154215 -0.49561142 -0.50310736]]
```

다중 분류에서는 `decision_function`의 결괏값 크기가 `(n_samples, n_classes)`입니다.
각 열은 각 클래스에 대한 확신 점수를 담고 있습니다. 수치가 크면 가능성이 큽니다.
전과 비슷하게 최댓값으로 예측을 할 수 있습니다.

``` python
print("가장 큰 결정 함수의 인덱스:\n{}".format(
      np.argmax(gbrt.decision_function(X_test), axis=1)))
print("예측:\n{}".format(gbrt.predict(X_test)))
# 가장 큰 결정 함수의 인덱스:
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0]
# 예측:
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0]
```

`predict_proba`의 출력값 크기도 같습니다. 마찬가지로 각 데이터 포인트에서 클래스 확률의 합은 1입니다.

``` python
# predict_proba 결과 중 앞부분 일부를 확인합니다
print("예측 확률:\n{}".format(gbrt.predict_proba(X_test)[:6]))
# 행 방향으로 확률을 더하면 1 이 됩니다
print("합: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))
# 예측 확률:
# [[0.10664722 0.7840248  0.10932798]
#  [0.78880668 0.10599243 0.10520089]
#  [0.10231173 0.10822274 0.78946553]
#  [0.10664722 0.7840248  0.10932798]
#  [0.10825347 0.66344934 0.22829719]
#  [0.78880668 0.10599243 0.10520089]]
# 합: [1. 1. 1. 1. 1. 1.]
```

여기서도 최댓값으로 예측을 할 수 있습니다.

``` python
print("가장 큰 예측 확률의 인덱스:\n{}".format(
      np.argmax(gbrt.predict_proba(X_test), axis=1)))
print("예측:\n{}".format(gbrt.predict(X_test)))
# 가장 큰 예측 확률의 인덱스:
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0]
# 예측:
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0]
```

클래스가 꼭 숫자인 것은 아니니 클래스만 `classes_`의 값을 사용하는 것이 좋습니다.
최종 결과값은 다음과 구할 수 있습니다. (책에서는 갑자기 로지스틱 회귀를 사용합니다. 바꿔치기..?)

``` python
logreg = LogisticRegression()
​
# iris 데이터셋의 타깃을 클래스 이름으로 나타내기
named_target = iris.target_names[y_train]
logreg.fit(X_train, named_target)
print("훈련 데이터에 있는 클래스 종류: {}".format(logreg.classes_))
print("예측: {}".format(logreg.predict(X_test)[:10]))
argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1)
print("가장 큰 결정 함수의 인덱스: {}".format(argmax_dec_func[:10]))
print("인덱스를 classses_에 연결: {}".format(logreg.classes_[argmax_dec_func][:10]))
# 훈련 데이터에 있는 클래스 종류: ['setosa' 'versicolor' 'virginica']
# 예측: ['versicolor' 'setosa' 'virginica' 'versicolor' 'versicolor' 'setosa'
#  'versicolor' 'virginica' 'versicolor' 'versicolor']
# 가장 큰 결정 함수의 인덱스: [1 0 2 1 1 0 1 2 1 1]
# 인덱스를 classses_에 연결: ['versicolor' 'setosa' 'virginica' 'versicolor' 'versicolor' 'setosa'
#  'versicolor' 'virginica' 'versicolor' 'versicolor']
```
