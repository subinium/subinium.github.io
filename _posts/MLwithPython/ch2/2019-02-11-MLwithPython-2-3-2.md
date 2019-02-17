---
title : \[ML with Python\] 2장. 지도 학습 - 지도 학습 알고리즘 (2)
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

2.3 지도 학습 알고리즘 (2)

> 본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

## 2.3.5 결정 트리

**결정 트리** 는 분류와 회귀 문제에 널리 사용하는 모델입니다. 기본적으로 결정 트리는 결정에 다다르기 위해 예/아니오 질문을 이어 나가면서 학습합니다. 마치 [아키네이터](https://en.akinator.com/) 같은 것입니다. 스무고개와 같은 과정으로 이어지는 것입니다.

### 결정 트리 만들기

우선 mglearn에서 제공하는 기본 이미지로 결정 트리에 대한 개념을 더 잡아봅시다. 코드는 생략하고 이미지만 가져오겠습니다.
예제에서 사용하는 데이터 셋은 다음과 같습니다. 각 클래스에는 50개의 데이터가 있습니다. 데이터가 후에 사용될 예정이며 데이터셋의 이름을 two_moons라고 합시다.

![21](https://i.imgur.com/lFv6tQz.png)

결정 트리를 학습한다는 것은 정답에 가장 빨리 도달하는 예/아니오 질문 목록을 학습하는 것입니다. 머신러닝에서는 이런 질문을 테스트라고 합니다.

트리를 만들 때 알고리즘은 가능한 모든 테스트에서 타깃값에 대해 가장 많은 정보를 가진 것을 고릅니다. 아래 그림은 첫 번째로 선택된 테스트를 보여줍니다.

![22](https://i.imgur.com/k0Lb6mY.png)

우선 여기서 첫 질문은 **루트 노드** 인 맨 위 노드에 존재하며, 모든 데이터를 테스트에 따라 나누게 됩니다.
좀 더 깊이가면 다음과 같이 만들어집니다.

![23](https://i.imgur.com/K2s6syQ.png)

반복된 프로세스는 각 노드가 테스트 하나씩을 가진 이진 결정 트리를 만듭니다. 계속 둘로 나누며 정확도를 늘립니다. 분할정복으로 볼 수 있습니다.  데이터를 분할하는 것은 각 분할된 영역이 한 개의 타깃값을 가질 때까지 반복됩니다. 타깃 하나로만 이뤄진 리프 노드를 **순수 노드** 라고 합니다. 이 데이터 셋 최종 분할 트리는 다음과 같습니다.

![24](https://i.imgur.com/BO9UNO0.png)

다음과 같은 방법으로 회귀 문제에도 트리를 사용할 수 있습니다. 예측을 하려면 각 노드의 테스트 결과에 따라 트리를 탐색해나가고 새로운 데이터 포인트에 해당되는 리프 노드를 찾습니다. 찾은 리프 노트의 훈련 데이터 평균값이 이 데이터 포인트의 출력이 됩니다.

### 결정 트리 복잡도 결정하기

일반적으로 모든 리프 노드가 순수 노드가 되기까지 진행하면 모델이 매우 복잡해지고 훈련 데이터에 과대적합됩니다.
순수 노드로 이루어진 트리는 훈련 세트에 100% 정확하다는 의미이니 과대적합은 거의 당연하게 나타납니다. 위에서 본 마지막 모델에서도 빨간 포인트 중 일부(이상치 하나)는 경향성과 맞지 않지만 잘 분류된 것을 확인할 수 있습니다.

결정 트리에서 과대적합을 막는 방법은 두 가지입니다.

1. **사전 가지치기(pre-prunning)** : 트리 생성을 일찍 중단하는 전략
2. **사후 가지치기(post-prunning)** : 트리를 만든 후 데이터 포인트가 적은 노드를 삭제하거나 병합하는 전략

사전 가지치기는 깊이 제한, 리프 개수 제한, 노드 분할에 따른 포인트 수 제한 등의 방법이 있습니다.
scikit-learn에서는 `DecisionTreeRegressor`와 `DecisionTreeClassifier`에 구현되어 있으며, 사전 가지치기만 지원합니다.

유방암 데이터셋을 사용하여 사전 가지치기의 효과를 확인해보겠습니다. 위에서와 마찬가지로 훈련, 테스트 세트로 나눕니다.
그리고 완전한 트리 모델과 가지치기한 모델을 비교해보겠습니다.

``` python
from sklearn.tree import DecisionTreeClassifier
​
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

# 훈련 세트 정확도: 1.000
# 테스트 세트 정확도: 0.937
```

완전한 트리 모델의 경우에는 기대대로 훈련 세트에서 정확도는 100%입니다.
테스트 세트의 정확도는 93.7%로 선형 모델 95%보다 살짝 낮습니다.

결정 트리의 깊이를 제한하지 않으면 깊이와 복잡도는 걷잡을 수 없이 커질 수 있습니다.
복잡도가 커지면 과대적합되기 쉬우니 이제 규제, 즉 제한이 필요합니다.
`DecisionTreeClassifier`에서는 `max_depth=4`과 같이 연속된 질문 수를 최대 4회로 제한할 수 있습니다.
훈련 세트의 정확도는 떨어질 수 있지만 테스트 세트의 성능을 개선할 수 있을 것입니다.

``` python
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
​
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))
# 훈련 세트 정확도: 0.988
# 테스트 세트 정확도: 0.951
```

예상대로 어느정도 과대 적합은 있지만, 테스트 세트 정확도가 개선된 것을 확인할 수 있습니다.
위 코드에서 `random_state=0`는 모델의 랜덤 성질을 같은 값으로 고정하기 위한 매개변수입니다.

### 결정 트리 분석

트리 모듈의 `export_graphviz`함수를 이용해 트리를 시각화할 수 있습니다. 이 함수는 그래프 저장용 텍스트 파일 포맷인 `.dot` 파일을 만듭니다. 각 노드에서 다수인 클래스를 색으로 나타내기 위해 옵션을 주고 적절히 레이블되도록 클래스 이름과 특성 이름을 매개변수로 전달합니다.

> **NOTE** `export_graphviz` 함수에 filled 매개변수를  `True`로 지정하면 노드의 클래스가 구분되도록 색이 칠해집니다.

``` python
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["악성", "양성"],
                feature_names=cancer.feature_names, impurity=False, filled=True)
```

이 파일을 읽어서 `graphviz` 모듈을 사용해 다음과 같이 시각화 해보겠습니다.

``` python
import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()

display(graphviz.Source(dot_graph))

# dot 파일 이미지로 저장하기
dot = graphviz.Source(dot_graph)
dot.format='png'
dot.render(filename='tree')
```

그냥 시각화할 경우 html 파일로만 작성되므로, 추가 명령어를 통해 png로 파일을 저장할 수 있습니다.

![tree](https://i.imgur.com/qqdFeDr.png)

트리를 시각화하면 알고리즘을 예측이 이뤄지는 과정을 볼 수 있고, 직관적인 구조를 볼 수 있습니다.
하지만 이진 트리이므로 깊이가 1씩 증가할 때마다, 그래프의 너비는 2배씩 커질 것입니다.
10 정도 깊이를 가면 그때는 시각화를 해도 분석하기 어려워질 수 있습니다.
각 노드는 다음과 같은 정보를 담고 있습니다.

- 조건문
- sample 수
- 악성/양성 수(1/0 또는 positive/negative)
- class

### 트리의 특성 중요도

위에서 말했듯이 깊이가 깊어지면 트리를 전체적으로 살피는 것은 어려울 수 있습니다.
대신 트리가 어떻게 작동하는지 요약하는 속성들을 사용할 수 있습니다.
가장 널리 사용되는 속성은 트리를 만드는 결정에 각 특성이 얼마나 중요한지를 평가하는 **특성 중요도(feature importance)** 입니다. 이 값은 0과 1 사이의 숫자로, 각 특성에 대해 0은 전혀 사용되지 않음을 1을 완벽하게 타깃 클래스를 예측했다는 뜻입니다. 그렇게 특성 중요도의 전체 합은 1입니다.

``` python
print("feature importance : \n {}".format(tree.feature_importances_))
# feature importance :
#  [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.01019737 0.04839825
#  0.         0.         0.0024156  0.         0.         0.
#  0.         0.         0.72682851 0.0458159  0.         0.
#  0.0141577  0.         0.018188   0.1221132  0.01188548 0.        ]
```

이 값을 막대 그래프로 시각화하면 더 명확하게 값에 대한 직관을 얻을 수 있습니다.

``` python
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature importance")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)
```

![25](https://i.imgur.com/qDHrrWi.png)

트리에서 첫번째 노드에서 사용되는 요소인 `worst radius` 값이 가장 크게 나타납니다. 이 요소가 두 클래스를 꽤나 잘 나누고 있음을 알 수 있습니다. 하지만 낮은 값이 유용하지 않다는 의미는 아닙니다. 단지 트리가 그 특성을 선택하지 않았을 뿐이며 다른 특성이 동일한 정보를 지니고 있어서일 수 있습니다.

선형 모델의 계수와는 다르게 요소가 클래스에 어떤 영향을 미치는지는 직접적으로 알 수 없습니다. 특성과 클래스 사이에는 간단하지 않은 관계가 있을 수 있습니다. 다항식과 다르게 미치는 영향이 그 전 질의에 따라서 지지하는 클래스가 쉽게 바뀔 수 있기 때문입니다.

### 회귀에서의 결정 트리

회귀에서도 결정 트리는 유용하게 사용할 수 있습니다.
`DecisionTreeRegressor`에서는 위에서 사용한 분류와 비슷하게 회귀 문제에 적용할 수 있습니다.
하지만 여기서는 **외삽(extrapolation)**, 즉 훈련 데이터의 범위 밖 포인트에 대해 예측을 할 수 없습니다.

컴퓨터 메모리 가격 동향 데이터셋을 이용해 더 자세히 살펴봅시다.
x축은 날짜, y축은 해당 년도의 램(RAM) 1메가바이트당 가격입니다.

``` python
import os
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))

# 로그 스케일로 그래프 그리기
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("years")
plt.ylabel("price ($/Mbyte)")
```

![26](https://i.imgur.com/RfCKT0I.png)

y축은 로그 스케일입니다. 그래프를 로그 스케일로 그리면 약간의 굴곡을 제외하고는 선형적으로 나타나서 예측이 쉬워집니다. (이 부분은 어떤 이유일지 궁금합니다. 데이터들간의 차이를 줄여주기 때문에 그런거 같기도 하고 ...)

여기서 날짜 특성 하나로 가격 예측을 시도해보겠습니다. 여기서는 `DecisionTreeRegressor`과 `LinearRegression`을 비교해봅니다. (딥러닝 측면에서 본다면 시계열 데이터이므로 LSTM 이나 1dConv를 사용하면 됩니다.)

`DecisionTreeRegressor`에서는 로그 스케일을 사용해도 아무런 차이가 없지만, `LinearRegression`에는 큰 차이가 있습니다.
모델을 훈련시키고 예측을 수행한 다음 로그 스케일을 돌리기 위해 지수 함수를 적용합니다.  그래프 표현을 위해 전체 데이터셋에 대해 예측을 수행합니다. (원래는 테스트 데이터셋과의 비교가 관심대상입니다.)

``` python
from sklearn.tree import DecisionTreeRegressor
# 2000년 이전을 훈련 데이터로, 2000년 이후를 테스트 데이터로 만듭니다
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]
​
# 가격 예측을 위해 날짜 특성만을 이용합니다
X_train = data_train.date[:, np.newaxis]
# 데이터와 타깃 사이의 관계를 간단하게 만들기 위해 로그 스케일로 바꿉니다
y_train = np.log(data_train.price)
​
tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)
​
# 예측은 전체 기간에 대해서 수행합니다
X_all = ram_prices.date[:, np.newaxis]
​
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)
​
# 예측한 값의 로그 스케일을 되돌립니다
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label="train data")
plt.semilogy(data_test.date, data_test.price, label="test data")
plt.semilogy(ram_prices.date, price_tree, label="tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="LR prediction")
plt.legend()
```

![27](https://i.imgur.com/nnsL9NY.png)

두 모델은 확실한 차이를 보이는 것을 알 수 있습니다.
선형 회귀는 그래도 실제 데이터와 비슷한 그래프를 그리는 반면에 결정 트리로 만든 결과는 미래 예측을 전혀하지 못하고 있습니다.
하지만 트리는 훈련 데이터에 대해서는 완벽하게 예측하고 있는 것을 확인할 수 있습니다. (트리 복잡도에 제한을 두지 않았으므로 알아서 하드코딩)

트리 모델은 훈련 데이터 밖의 새로운 데이터를 예측할 능력이 없습니다. 이는 모든 트리 기반 모델의 공통된 단점입니다.

> **NOTE** 트리 기반 모델도 좋은 예측을 만들 수 있습니다. 이 예제의 목적은 트리 모델이 시계열 데이터에는 잘 맞지 않는다는 것과 트리가 어떻게 예측을 만드는지 그 특성을 보여주기 위함입니다. 가격이 오를 것인가 내릴 것인가와 같은 문제에 대해서는 모델의 모양과 특성에 따라 예측이 가능합니다.

### 장단점과 매개변수

결정 트리에서 모델 복잡도를 조절하는 매개변수는 트리가 완전히 만들어지기 전에 멈추게 하는 사전 가지치기 매개변수입니다.

결정 트리가 이전에 소개한 다른 알고리즘들보다 나은 점은 두 가지입니다.

1. 만들어진 모델을 쉽게 시각화할 수 있습니다. 이는 비전문가에게 설명하기 쉽고, 이해하기 쉽습니다.
2. 데이터 스케일의 영향을 받지 않습니다. 그러므로 전처리(정규화, 표준화)가 필요 없습니다.

주요 단점은 과대적합되는 경향이 있어 일반화 성능이 좋지 않다는 것입니다. 그래서 다음에 설명할 앙상블 방법을 단일 결정 트리의 대안으로 흔히 사용합니다.

## 2.3.6 결정 트리의 앙상블

**앙상블(ensemble)** 은 여러 머신러닝 모델을 연결하여 더 강력한 모델을 만드는 기법입니다.
머신러닝에는 이런 종류의 모델이 많지만, 그중에서도 두 앙상블 모델이 분류와 회귀 문제의 다양한 데이터셋에서 효과적이라고 입증되었습니다. **랜덤 포레스트(random forest)** 와 **그래디언트 부스팅(gradient boosting)** 결정 트리는 둘 다 모델을 구성하는 기본 요소로 결정 트리를 사용합니다.

### 랜덤 포레스트

결정 트리의 가장 큰 단점은 과대적합입니다. 랜덤 포레스트는 이 문제를 회피할 수 있는 방법입니다.
랜덤 포레스트는 기본적으로 조금씩 다른 여러 결정 트리의 묶음입니다.

잘 작동하되 서로 다른 다른 방향으로 과대적합된 트리를 많이 만들면 그 결과의 평균을 사용하여 과대적합 양을 줄이는 것이 메인 아이디어입니다. 이렇게 하면 트리 모델의 예측 성능이 유지되면서 과대적합이 줄어드는 것이 수학적으로 증명되었습니다.

그렇다면 이제 해결해야하는 문제는 2개입니다. 잘 작동하는 결정 트리를 많이 만드는 것과 그것들의 방향성이 다양해야한다는 것입니다.
랜덤 포레스트는 이름에서도 알 수 있듯이 트리를 랜덤으로 만듭니다. 랜덤으로 만드는 방법은 두 가지입니다.

- 트리를 만들 때 사용하는 데이터 포인트를 무작위로 선택
- 분할 테스트에서 특성을 무작위로 선택

### 랜덤 포레스트 구축

랜덤 포레스트 모델을 만들려면 생성할 트리 개수를 정해야 합니다. (`RandomForestRegressor`나 `RandomForestClassifier`에서의 `n_estimators` 매개변수, 여기서는 10으로 가정)
트리들은 완전히 독립적으로 만들어져야 하므로 알고리즘은 각 트리가 고유하게 만들어지도록 무작위한 선택을 합니다.
트리를 만들기 위해 먼저 데이터의 **부트스트랩 샘플(boostrap sample)** 을 생성합니다.
n_samples개의 데이터 포인트 중 무작위로 데이터를 n_samples 횟수만큼 반복 추출합니다. 원래 데이터셋과 크기는 같지만 중복때문에 1/3 정도의 데이터가 누락됩니다. (정확히는 $\frac{1}{e}$)

그리고 이렇게 만든 데이터셋으로 결정 트리를 만듭니다. 하지만 전과 같이 최신의 테스트를 찾는 것이 아닌 알고리즘이 각 노드에서 후보 특성을 무작위로 선택한 후 이 후보들 중에서 최선의 테스트를 찾습니다. 특성을 고르는 것은 `max_features` 매개변수로 조정할 수 있습니다.
후보 특성을 고르는 것은 매 노드마다 반복되므로 트리의 각 노드는 다른 후보 특성들을 사용하여 테스트를 사용하여 테스트를 만듭니다.

- 부트스트랩 샘플링은 랜덤 포레스트 트리가 조금씩 다른 데이터셋을 이용해 만들어지도록합니다.
- 각 노드에서 특성의 일부만 사용하기 때문에 트리의 각 분기는 각기 다른 특성 부분 집합을 사용합니다.

이 두 메커니즘이 합쳐서 랜덤 포레스트의 모든 트리가 서로 달라지도록 만듭니다.

이 방식에서 핵심은 `max_features` 입니다. `max_features`를 `n_features`로 설정하면 모든 특성을 고려하므로 특성 선택에서의 무작위성이 들어가지 않습니다. 그렇다면 `max_features`을 1로 한다면 어떻게 될까요?
노드는 특성을 고려할 필요없이 각 데이터에 대한 임계값만 구하면 됩니다.

랜덤 포레스트로 예측을 할 때는 먼저 알고리즘이 모델에 있는 모든 트리의 예측을 만듭니다.
회귀에서는 이 예측의 평균값이 최종 예측입니다.
분류에서는 약한 투표 전략을 사용합니다. 각 알고리즘이 가능성 있는 출력 레이블의 확률을 제공함으로써 간접적인 예측을 합니다.
예측한 확률을 평균 내어 가장 높은 확률을 가진 클래스가 예측값이 됩니다.

### 랜덤 포레스트 분석

이전에 사용했던 데이터셋 two_moons에서 5개의 트리를 이용한 랜덤 포레스트를 만들어봅시다.

``` python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
​
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
​
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
```

이제 이를 시각화해봅시다.

``` python
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)

mglearn.plots.plot_2d_separator(forest, X, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
```

![28](https://i.imgur.com/0WFggem.png)

그림만 봐도 일반화가 잘된 것을 알 수 있습니다. 다섯 개의 결정 트리의 모양도 각기 다를 뿐만 아니라 랜덤 포레스트의 결과는 일반성을 가지는 것을 확인할 수 있습니다. 실제 어플리케이션에서는 더 많은 트리를 사용하므로 훨씬 부드러운 결정 경계가 만들어집니다.


이제 유방암 데이터셋에 100개의 트리로 이뤄진 랜덤 포레스트를 적용해보겠습니다.

``` python
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
​
print("train set acc: {:.3f}".format(forest.score(X_train, y_train)))
print("test set acc: {:.3f}".format(forest.score(X_test, y_test)))
# train set acc: 1.000
# test set acc: 0.972
```

단일 결정 트리보다 높은 97% 정확도를 가지는 것을 확인할 수 있습니다. 단일 결정 트리에서와 같은 방법으로 `max_features` 매개변수를 조정하거나 가지치기를 할 수도 있지만, 기본 설정으로도 좋은 결과를 얻을 수 있습니다.

랜덤 포레스트도 특성 중요도를 제공하는데, 각 트리의 특성 중요도를 취합한 값입니다. 더 신뢰할 수 있는 값입니다. 이전에 만든 함수를 이용한 그래프는 다음과 같습니다.

![29](https://i.imgur.com/NvlF1zP.png)

역시 마찬가지로 `worst radius`를 중요한 특성으로 선택했지만, 그보다 `worst perimeter`를 더 많이 사용한 특성으로 선택했습니다. 랜덤 포레스트를 만드는 무작위성은 알고리즘이 가능성 있는 많은 경우를 고려할 수 있도록 하므로, 그 결과 더 넓은 시각으로 데이터를 바라볼 수 있습니다.

### 장단점과 매개변수

회귀와 분류에 있어서 랜덤 포레스트는 현재 가장 널리 사용되는 머신러닝 알고리즘입니다.
성능이 뛰어나고 매개변수 튜닝을 많이 하지 않아도 잘 작동하며 데이터의 스케일을 맞춰줄 필요도 없습니다.

트리를 많이 만드는 만큼 시간이 많이 걸릴 수 있습니다.
그때는 `n_jobs` 매개변수를 이용해 사용하는 CPU를 늘려 병렬처리 해야합니다. CPU 수에 비례하여 빨라집니다.
기본값은 1이며 -1로 설정할 경우 모든 CPU를 사용합니다.

랜덤 포레스트는 랜덤에 기초하기에 같은 결과를 여러번 내기 위해서는 `random_state`값을 고정해야합니다.
그렇기에 모든 예제는 `random_state`값을 설정합니다. 트리가 많다면 차이가 줄겠지만 같은 값을 위해서는 고정이 필요합니다.

랜덤 포레스트는 텍스트 데이터와 같이 매우 차원이 높고 희소한 데이터에는 잘 작동하지 않으므로 그 때는 선형 모델을 사용해야합니다.
선형 모델보다 느리고, 메모리도 많이 사용하므로 필요에 따라 사용하는 모델을 선택하는 것이 중요합니다.

중요 매개변수는 `n_estimators`, `max_features`이고  `max_depth` 같은 사전 가지치기 옵션이 있습니다.
`n_estimators`는 클수록 성능이 좋습니다. 메모리와 훈련 시간의 비용이 커지는 단점이 있습니다.

`max_features`는 각 트리가 얼마나 무작위가 될지를 결정하며 작은 값은 과대적합을 줄여줍니다. 일반적으로는 기본값을 사용하는 것을 추천합니다.
분류는 `sqrt(n_features)`, 회귀는 `n_features`를 사용합니다.

### 그래디언트 부스팅 회귀 트리

그래디언트 부스팅 회귀 트리는 여러 개의 결정 트리를 묶어 강력한 모델을 만드는 또 다른 앙상블 방법입니다.
이름은 회귀지만 이 모델은 회귀와 분류 모두에 사용할 수 있습니다.

랜덤 포레스트와 다르게 그래디언트 부스팅은 이전 트리의 오차를 보완하는 방식으로 순차적으로 트리를 만듭니다.
무작위성이 없고, 강력한 사전 가지치기를 사용한다는 특징이 있습니다.
보통 하나에서 다섯 정도의 깊지 않은 트리를 사용하므로 메모리를 적게 사용하고 예측도 빠릅니다.
그래디언트 부스팅의 근본 아이디어는 이런 얕은 트리 같은 간단한 모델(**약한 학습기(weak learner)**)을 많이 연결하는 것입니다.
각각의 트리는 데이터 일부에 대해서만 예측을 잘 수행할 수 있어서 트리가 많이 추가될수록 성능이 좋습니다.

> **NOTE** 그래디언트 부스팅은 이전에 만든 트리의 예측과 타깃값 사이의 오차를 줄이는 방향으로 새로운 트리를 추가하는 알고리즘입니다. 이를 위해 손실 함수를 정의하고 경사 하강법을 사용하여 다음에 추가될 트리가 예측해야 할 값을 보정해나갑니다.

머신러닝 경연 대회에서 우승을 많이 차지한 알고리즘이며 업계에서도 널리 사용합니다. 랜덤 포레스트보다 매개변수 설정에 민감하지만 잘 조정하면 높은 정확도를 제공합니다.

앙상블 방식에 있는 사전 가지치기나 트리 개수 외에도 그래디언트 부스팅에서 중요한 매개변수는 이전 트리의 오차를 얼마나 강하게 보정할  것인지를 제어하는 `learning_rate`입니다. 학습률이 크면 보정을 강하게 하므로 복잡한 모델이 만들어집니다.
`n_estimators` 값을 키우면 앙상블에 트리가 많이 추가되어 모델의 복잡도가 커지고 훈련 세트에서 실수를 바로잡을 기회가 더 많아집니다.

아래는 유방암 데이터셋을 이용해 `GradientBoostingClassifier`를 사용한 예입니다. 기본값인 깊이가 3인 트리 100개와 학습률 0.1을 사용했습니다.

``` python
from sklearn.ensemble import GradientBoostingClassifier
​
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
​
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
​
print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))
# 훈련 세트 정확도: 1.000
# 테스트 세트 정확도: 0.958
```

기본값에서는 훈련 세트에서 100%인것으로 보아 과대적합되어 있음을 알 수 있습니다.
최대 깊이를 줄이거나 사전 가지치기를 강하게 하거나 학습률을 낮출 수 있습니다.

``` python
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
​
print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))
# 훈련 세트 정확도: 0.991
# 테스트 세트 정확도: 0.972
```
``` python
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
​
print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))
# 훈련 세트 정확도: 0.988
# 테스트 세트 정확도: 0.965
```

학습률을 낮추는 것보다는 깊이를 낮추는 것으로 모델 성능을 향상 시켰습니다.
다른 결정 트리 기반의 모델처럼 특성 중요도를 시각화해보겠습니다.

``` python
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

plot_feature_importances_cancer(gbrt)
```

![30](https://i.imgur.com/aBYJirS.png)

그래디언트 부스팅의 경우에는 일부 특성을 완전 무시하는 것을 확인할 수 있습니다.
비슷한 종류의 데이터에서 그래디언트 부스팅과 랜덤 포레스트 둘 다 잘 작동하지만, 보통 안정적인 랜덤 포레스트를 먼저 사용합니다.
랜덤 포레스트가 잘 작동하더라도 시간이 중요하거나 성능의 끝을 확인하고 싶을 때 그래디언트 부스팅을 사용하면 도움이 됩니다.

대규모 머신러닝 문제에 그래디언트 부스팅을 적용하려면 xgboost 패키지와 파이썬 인터페이스를 검토해보는 것을 추천합니다.

### 장단점과 매개변수

그래디언트 부스팅 결정 트리는 지도 학습에서 가장 강력하고 널리 사용되는 모델 중 하나입니다. 가장 큰 단점은 매개변수를 잘 조정해야 한다는 것과 훈련 시간이 길다는 것입니다. 다른 트리 기반 모델처럼 스케일 조정이 필요 없고, 이전 특성이나 연속적인 특성에서 잘 작동합니다. 그리고 트리 기반 모델의 특성상 희소한 고차원 데이터에는 잘 작동하지 않습니다.

중요 매개변수는 트리의 개수를 설정하는 `n_estimators`와 학습률 `learning_rate`입니다.
일반적인 관례는 가용한 시간과 메모리 한도에서 `n_estimators`를 맞추고 나서 `learning_rate`를 찾는 것입니다.

중요한 또 다른 매개변수는 각 트리의 복잡도를 낮추는 `max_depth`(또는 `max_leaf_nodes`)입니다. 매우 작게 설정하며 트리의 깊이가 5보다 깊어지지 않게 합니다.

## 2.3.7 커널 서포트 벡터 머신

이전에 *분류용 선형 모델* 에서 선형 서포트 벡터 머신을 사용해 분류 문제를 풀어보았습니다.
커널 서포트 벡터 머신(보통 SVM이라 부르는)은 입력 데이터에서 단순한 초평면으로 정의되지 않는 더 복잡한 모델을 만들 수 있도록 확장한 것입니다. 서포트 벡터 머신을 분류와 회귀에 모두 사용할 수 있지만 여기서는 SVC를 사용하는 분류 문제만을 다루겠습니다.
SVR를 사용하는 회귀 문제에도 같은 개념을 적용할 수 있습니다.

### 선형 모델과 비선형 특성

직선과 초평면은 유연하지 못하여 저차원 데이터셋에서는 선형 모델이 매우 제한적입니다. 선형 모델을 유연하게 만드는 한 가지 방법은 특성끼리 곱하거나 특성을 거듭제곱하는 식으로 새로운 특성을 추가하는 것입니다.

인위적 데이터에서 살펴봅시다.

``` python
X, y = make_blobs(ceneters=4, random_state=8)
y = y % 2

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
```

![31](https://i.imgur.com/cWOGWUY.png)

다음 데이터는 선형적으로는 구분되지 않습니다. 분류를 위한 선형 모델은 직선으로만 데이터 포인트를 나눌 수 있어서 이런 데이터셋에는 잘 들어맞지 않습니다.

``` python
from sklearn.svm import LinearSVC
linear_svm = LinearSVC(max_iter=10000).fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
```

![32](https://i.imgur.com/hca13Ow.png)

두 번째 특성을 제곱한 특성 1 ** 2 를 추가해 특성을 확장해보겠습니다.
이제 2차원에서 3차원으로 데이터가 확장되었습니다. 이 데이터셋을 3차원 산점도로 나타내봅시다.

``` python
# 두 번째 특성을 제곱하여 추가합니다
X_new = np.hstack([X, X[:, 1:] ** 2])

from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
# 3차원 그래프
ax = Axes3D(figure, elev=-152, azim=-26)
# y == 0 인 포인트를 먼저 그리고 그 다음 y == 1 인 포인트를 그립니다
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.set_xlabel("f0")
ax.set_ylabel("f1")
ax.set_zlabel("f1 ** 2")
ax.legend()
plt.show()
```
![33](https://i.imgur.com/8POl2sj.png)

3차원으로 보니 평면으로 분할해야할 부분이 대충 감으로 옵니다. 이를 그래프에 다시 그려봅시다.
(이 전까지는 소스코드가 대부분 이해하며 했는데, 확실히 matplotlib 응용은 어렵군요...)

``` python
linear_svm_3d = LinearSVC(max_iter=10000).fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

# 선형 결정 경계 그리기
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')

ax.set_xlabel("f0")
ax.set_ylabel("f1")
ax.set_zlabel("f1 ** 2")
```
![34](https://i.imgur.com/pR7E21b.png)

3차원에서는 평면으로 분류할 수 있지만 원래 특성으로 투영하면 더 이상 선형이 아닙니다. 직선보다 타원에 가까운 모습인 것을 확인할 수 있습니다.

``` python
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
             cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
```

![35](https://i.imgur.com/qA0cvwE.png)

### 커널 기법

앞에서 비선형 특성을 추가하여 선형 모델을 강력하게 만들어보았습니다. 하지만 많은 특성의 경우 어떤 특성을 추가해야 할지 모르고 특성을 많이 추가하면 연산 비용이 커집니다. 다행히 수학적 기교를 사용해서 새로운 특성을 많이 만들지 않고서도 고차원에서 분류기를 학습시킬 수 있습니다. 이를 **커널 기법(kernel trick)** 이라고 하며 실제로 데이터를 확장하지 않고 확장된 특성에 대해 데이터 포인트들의 거리를 계산합니다. (스칼라 곱)

SVM에서 데이터를 고차원 공간에 매칭하는 데 많이 사용하는 방법은 두 가지입니다. 원래 특성 가능한 조합을 지정된 차수까지 모두 계산하는 다항식 커널부터, 가우시안 커널로 불리는 RBF 커널이 있습니다.  

### SVM 이해하기

학습이 진행되는 동안 SVM은 각 훈련 데이터 포인트가 투 클래스 사이의 결정 경계를 구분하는 데 얼마나 중요한지를 배우게 됩니다.
일반적으로 훈련 데이터의 일부만 결정 경계를 만드는 데 영향을 줍니다. 바로 두 클래스 사이의 경계에 위치한 데이터 포인트들입니다.
이런 데이터 포인트를 **서포트 벡터(support vector)** 라고 하며, 여기서 서포트 벡터 머신이란 이름이 유래했습니다.

새로운 데이터 포인트에 대해 예측하려면 각 서포트 벡터와의 거리를 측정합니다.
분류 결정은 서포트 벡터까지 거리에 기반하며 서포트 벡터의 중요도는 훈련 과정에서 학습합니다.(SVC 객체의 `dual_coef_` 속성)

데이터 포인트 사이의 거리는 가우시안 커널에 의해 계산됩니다.

$$k_{rbf}(x_1, x_2) = exp(-\gamma \lVert x_1 - x_2 \rVert^2)$$

여기서에 $x_1$, $x_2$는 데이터 포인트 이며, $\lVert$는 유클리디안 거리를 의미하고, $\gamma$는 가우시안 커널의 폭을 제어하는 매개변수입니다.

두 개의 클래스를 가진 2차원 데이터셋에 서포트 벡터 머신을 학습시켜보겠습니다. 결정 경계는 검은 실선으로, 서포트 벡터는 굵은 테두리로 크게 그렸습니다. forge 데이터셋을 SVM을 학습시켜 그래프를 그려보겠습니다.

``` python
from sklearn.svm import SVC

X, y = mglearn.tools.make_handcrafted_dataset()      
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
# 데이터 포인트 그리기
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 서포트 벡터
sv = svm.support_vectors_
# dual_coef_ 의 부호에 의해 서포트 벡터의 클래스 레이블이 결정됩니다
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
```

![36](https://i.imgur.com/4KwPvhp.png)

이 그림에서 SVM은 매우 부드럽고 비선형 경계를 만들었습니다.
여기서 사용한 매개변수 `C`와 `gamma`에 대해 알아보겠습니다.

### SVM 매개변수 튜닝

`gamma`는 앞 절 공식에 나와있는 $\gamma$로 가우시안 커널 폭의 역수에 해당합니다.
`gamma` 매개변수가 하나의 훈련 샘플이 미치는 영향의 범위를 결정합니다. 작은 값은 넓은 영역을 뜻하며 큰 값이라면 영향이 미치는 범위가 제한적입니다. (공식에 -가 붙어있기 때문입니다.)

`C`매개변수는 선형 모델에서 사용한 것과 유사한 규제 매개변수입니다. 이 매개변수는 각 포인트의 중요도(`dual_coef_`값)을 제한합니다. 매개변수에 따른 변경을 살펴보겠습니다.

``` python
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

axes[0, 0].legend(["class 0", "class 1", "class 0 SV", "class 1 SV"], ncol=4, loc=(.9, 1.2))
```

![37](https://i.imgur.com/dAFtY25.png)

왼쪽에서 오른쪽으로 가면서 `gamma` 매개변수를 0.1에서 10으로 증가시켰습니다.
작은 `gamma`값은 가우시안 커널의 반경을 크게 하여 많은 포인트들이 가까이 있는 것으로 고려됩니다.
그렇기에 왼쪽에서 오른쪽으로 갈수록 하나의 포인트에 대해 민감해지는 것을 확인할 수 있습니다.
다르게 말하면 모델이 복잡해진다는 의미입니다.

위에서 아래로는 `C` 매개변수를 0.1에서 1000으로 증가시켰습니다.
선형 모델에서처럼 작은 C는 매우 제약이 큰 모델을 만들고 각 데이터 포인트의 영향력이 작습니다.
왼쪽 위의 결정 경계는 거의 선형에 가까우며 잘못 분류된 데이터 포인트가 주는 영향이 거의 없습니다.
하지만 매개변수를 크게 한 왼쪽 아래에서는 결정 경계를 휘어서 데이터를 확실하게 분류하는 것을 확인할 수 있습니다.

RBF 커널 SVM을 유방암 데이터셋에 적용해보겠습니다. 기본갑 `C=1, gamma=1/n_features`를 사용합니다.

``` python
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)

# print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train, y_train)))
# print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test, y_test)))
```

훈련세트에서는 완벽한 점수를 냈지만 테스트 세트는 63% 정확도를 가집니다. 이제 이정도는 과대적합임을 아실 것입니다.
SVM은 잘 작동하는 편이지만 매개변수 설정과 데이터 스케일에 매우 민감합니다.
특히 입력 특성의 범위가 비슷해야합니다. 각 특성의 최솟값과 최댓값을 로그 스케일로 나타내보겠습니다.

``` python
plt.boxplot(X_train, manage_xticks=False)
plt.yscale("symlog")
plt.xlabel("feature list")
plt.ylabel("feature size")
```

![38](https://i.imgur.com/JItQpm3.png)

데이터에 따라 스케일이 매우 다른 것을 알 수 있습니다. SVM에서는 이런 스케일 차이는 매우 치명적입니다.
이제 이를 위한 전처리를 알아봅시다.

### SVM을 위한 데이터 전처리

커널 SVM에서는 모든 특성 값을 0에서 1사이로 맞추는 방법을 많이 사용합니다. 후에 MinMaxScaler 전처리 메서드를 사용해서 할 수 있지만 여기서는 직접 만들어봅시다.

``` python

# 훈련 세트에서 특성별 최솟값 계산
min_on_training = X_train.min(axis=0)
# 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
range_on_training = (X_train - min_on_training).max(axis=0)
​
# 훈련 데이터에 최솟값을 빼고 범위로 나누면
# 각 특성에 대해 최솟값은 0 최댓값은 1 임
X_train_scaled = (X_train - min_on_training) / range_on_training
print("특성별 최솟값\n{}".format(X_train_scaled.min(axis=0)))
print("특성별 최댓값\n {}".format(X_train_scaled.max(axis=0)))
# 특성별 최솟값
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0.]
# 특성별 최댓값
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1.]

# 테스트 세트에도 같은 작업을 적용하지만
# 훈련 세트에서 계산한 최솟값과 범위를 사용합니다(자세한 내용은 3장에 있습니다)
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC(gamma='auto')
svc.fit(X_train_scaled, y_train)
​
print("훈련 세트 정확도: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("테스트 세트 정확도: {:.3f}".format(svc.score(X_test_scaled, y_test)))
# 훈련 세트 정확도: 0.948
# 테스트 세트 정확도: 0.951

svc = SVC(gamma='auto', C=1000)
svc.fit(X_train_scaled, y_train)
​
print("훈련 세트 정확도: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("테스트 세트 정확도: {:.3f}".format(svc.score(X_test_scaled, y_test)))
# 훈련 세트 정확도: 0.988
# 테스트 세트 정확도: 0.972
```

데이터 스케일을 조정하고 적당하게 매개변수를 설정한 결과 97%까지 성능이 향상된 것을 확인할 수 있습니다.

### 장단점과 매개변수

커널 서포트 벡터 머신은 강력한 모델이며 다양한 데이터셋에서 잘 작동합니다.
SVM은 데이터의 특성이 몇 개 안되더라도 복잡한 결정 경계를 만들 수 있습니다.
저차원과 고차원 데이터에 모두 잘 작동하지만 샘플이 많을 때는 잘 맞지 않습니다.
10000개 정도의 샘플에는 괜찮지만, 100000개 이상의 데이터셋에서는 메모리, 속도 관점에서 도전적인 과제입니다.

SVM의 또 다른 단점은 데이터 전처리와 매개변수 설정에 신경을 많이 써야 한다는 점입니다.
그렇기에 랜덤 포레스트나 그래디언트 부스팅 같은 트리 기반 모델을 애플리케이션에 많이 사용합니다.
하지만 모든 특성이 비슷한 단위이고 스케일이 비슷하면 SVM은 좋은 시도입니다.


커널 SVM에서 중요한 매개변수는 규제 매개변수 `C`이고 어떤 커널을 사용할지와 각 커널에 따른 매개변수입니다.
RBF에서는 `gamma` 매개변수를 사용했으며, 이외에도 다양한 커널과 매개변수가 존재합니다.

## 2.3.8 신경망(딥러닝)

본 절은 이미 [deeplearning with python](/Keras)에서 언급 하였기에 생략합니다.
후에 필요하다고 생각되면 추가하겠습니다.
