---
title : \[ML with Python\] 3장 비지도 학습과 데이터 전처리 - 차원 축소, 특성 추출, 매니폴드 학습
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

3.4 차원 축소, 특성 추출, 매니폴드 학습

> 본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

## 3.4.0 INTRO

데이터 변환의 이유는 여러 가지입니다.
가장 일반적인 동기는 시각화하거나, 데이터를 압축하거나, 추가적인 처리를 위해(지도 학습) 정보가 더 잘 드러나는 표현을 찾기 위해서입니다.
이번 절에서는 다음과 같은 알고리즘을 소개합니다.

- 주성분 분석 : 가장 간단하고 흔히 사용
- 비음수 행렬 분석 : 특성 추출
- t-SNE : 2차원 산점도를 이용해 시각화 용도로 사용

## 3.4.1 주성분 분석(PCA)

**주성분 분석(principal component analysis, PCA)** 은 통계적으로 상관관계가 없도록 데이터셋을 회전시기는 기술입니다.
단순히 회전만 하는 것은 아니고 회전을 한 뒤에 데이터를 설명하는데 필요한 (새로운)특성 중 일부만 선택합니다.
다음 예제로 그 의미를 확실히 할 수 있습니다.

![3](https://i.imgur.com/dRAtKnq.png)

첫 번째 그래프는 원본 데이터 포인트를 색으로 구분해 표시한 것입니다. 이 알고리즘은 먼저 **성분 1** 이라고 쓰인 분산이 가장 큰 방향을 찾습니다.
이 방향이 데이터에서 가장 많은 정보를 담고 있는 방향입니다. 특성들의 상관관계가 큰 방향입니다.
그리고 첫 번째 방향과 직각인 방향 중에서 가장 많은 정보를 담은 방향을 찾습니다.
2차원에서는 하나지만 고차원에서는 무한히 많은 직각 방향이 있습니다.

화살표의 방향은 중요하지 않습니다. 반대로 그려도 괜찮다는 것입니다. :-)
그렇게 찾은 방향을 데이터에 있는 주된 분산 방향이라고 하여 **주성분(principal component)** 라고 합니다. 일반적으로 원본 특성 개수만큼의 주성분이 있습니다. (기저 벡터의 개념으로 생각하면 됩니다.)

두 번째 그래프는 같은 데이터지만 주성분 1과 2를 각각 x축과 y축에 나란하도록 회전한 것입니다. 회전하기 전에 데이터에서 평균을 빼서 원점에 맞췄습니다. PCA에 의해 회전된 두 축은 연관되어 있지 않으므로 변환된 데이터에서 상관관계 행렬이 대각선 방향을 제외하고는 0이 됩니다.

PCA는 주성분의 일부만 남기는 차원 축소 용도로 사용할 수 있습니다. 이 예에서는 왼쪽 아래 그림과 같이 첫 번째 주성분만 유지하려고 합니다. 그렇게 되면 2차원 -> 1차원으로 차원이 감소합니다. 가장 유용한 성분을 남기는 것입니다.

마지막으로 데이터에 다시 평균을 더하고 반대로 회전시킵니다. 이 결과가 마지막 그래프입니다.
데이터 포인터들은 원래 특성 공간에 놓여 있지만 첫 번째 주성분의 정보만 담고 있습니다.
이 변환은 ***데이터에서 노이즈를 제거하거나 주성분에서 유지되는 정보를 시각화*** 하는 데 종종 사용합니다.

### PCA를 적용해 유방암 데이터셋 시각화하기

PCA가 가장 널리 사용되는 분야는 고차원 데이터셋의 시각화입니다.
세 개 이상의 특성을 가진 데이터를 산점도로 표현하는 것은 어렵습니다.
cancer 데이터셋에는 특성을 30개나 가지고 있어, 산점도를 그리기 위해서는 435개의 산점도를 그려야합니다. ($_{30}C_{2}$)
이보다 쉬운 방법은 양성과 악성 두 클래스에 대해 각 특성의 히스토그램을 그리는 것입니다.

![4](https://i.imgur.com/DwrZtc3.png)

이 그림은 각 특성에 대한 히스토그램으로 특정 간격(bin)에 얼마나 많은 데이터 포인트가 나타나는지 횟수를 센 것입니다.
각 그래프는 히스토그램 두 개를 겹쳐놓은 것으로 초록색은 양성 클래스의 포인트, 푸른색은 음성 클래스의 포인트입니다.

그래프를 보면 어떤 특성이 차이가 나고, 어떤 특성이 영향이 큰지 정도는 감이 옵니다.
하지만 특성 간의 상호작용이나 이 상호작용이 클래스와 어떤 관련이 있는지는 전혀 알려주지 못합니다.
PCA를 이용하면 주요 상호작용을 찾아낼 수 있어 더 나은 그림을 만들 수 있습니다.

이제 스케일을 조정하고 PCA를 활용해 산점도를 그려보겠습니다.
우선 `StandardScaler`를 사용해 각 특성의 분산이 1이 되도록 조정하겠습니다.

``` python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
```

PCA 변환을 학습하고 적용하는 것은 전처리만큼 간단합니다.

1. PCA 객체 생성
2. `fit` 메서드를 호출해 주성분 찾기
3. `transform` 메서드를 호출해 데이터 호출시키고 차원 축소하기

기본값일 때는 회전과 이동만 하니, 차원을 줄이려면 매개변수로 전달해야 합니다.

``` python
from sklearn.decomposition import PCA
# 데이터의 처음 두 개 주성분만 유지시킵니다
pca = PCA(n_components=2)
# 유방암 데이터로 PCA 모델을 만듭니다
pca.fit(X_scaled)

# 처음 두 개의 주성분을 사용해 데이터를 변환합니다
X_pca = pca.transform(X_scaled)
print("원본 데이터 형태: {}".format(str(X_scaled.shape)))
print("축소된 데이터 형태: {}".format(str(X_pca.shape)))

# 원본 데이터 형태: (569, 30)
# 축소된 데이터 형태: (569, 2)
```

그리고 맨 처음 두 개의 주성분을 이용해 산점도를 그립니다.

``` python
# 클래스를 색깔로 구분하여 처음 두 개의 주성분을 그래프로 나타냅니다.
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(["negative", "positive"], loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("1st pc")
plt.ylabel("2nd pc")
```

![5](https://i.imgur.com/WnAafLV.png)

PCA는 비지도 학습이므로 회전축을 찾을 때 어떤 클래스 정보도 사용하지 않습니다. 단순히 데이터에 있는 상관관계만 고려합니다.
두 클래스가 2차원 공간에서 꽤 잘 구분되는 것을 볼 수 있습니다.
이런 그림이면 선형 분류기로도 두 클래스를 잘 구분할 수 있을 것 같습니다.

하지만 PCA의 단점 중 하나는 그래프의 두 축을 해석하기 쉽지 않다는 점입니다.
여러 특성이 조합된 형태인 것만 알고있습니다. 이에 대한 정보는 PCA 객체가 학습될때(`fit` 메서드가 호출될 때입니다.) `components_` 속성에 저장됩니다.

``` python
print("PCA 주성분 형태: {}".format(pca.components_.shape))
# PCA 주성분 형태: (2, 30)
```

`components_`의 각 행은 주성분 하나씩을 나타내며 중요도에 따라 정렬되어 있습니다.
열은 원본 데이터의 특성에 대응하는 값입니다.

``` python
print("PCA 주성분: {}".format(pca.components_))
# PCA 주성분: [[ 0.21890244  0.10372458  0.22753729  0.22099499  0.14258969  0.23928535
#    0.25840048  0.26085376  0.13816696  0.06436335  0.20597878  0.01742803
#    0.21132592  0.20286964  0.01453145  0.17039345  0.15358979  0.1834174
#    0.04249842  0.10256832  0.22799663  0.10446933  0.23663968  0.22487053
#    0.12795256  0.21009588  0.22876753  0.25088597  0.12290456  0.13178394]
#  [-0.23385713 -0.05970609 -0.21518136 -0.23107671  0.18611302  0.15189161
#    0.06016536 -0.0347675   0.19034877  0.36657547 -0.10555215  0.08997968
#   -0.08945723 -0.15229263  0.20443045  0.2327159   0.19720728  0.13032156
#    0.183848    0.28009203 -0.21986638 -0.0454673  -0.19987843 -0.21935186
#    0.17230435  0.14359317  0.09796411 -0.00825724  0.14188335  0.27533947]]
```

이를 히트맵으로 시각화하면 이해하는데 용이합니다.

``` python
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["lst pc", "2nd pc"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("feature")
plt.ylabel("pc")
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["lst pc", "2nd pc"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("feature")
plt.ylabel("pc")
```

![6](https://i.imgur.com/QRAxEQU.png)

### 고유얼굴(eigenface) 특성 추출

PCA는 특성 추출에도 이용합니다. 원본 데이터 표현보다 분석하기에 더 적합한 표현을 찾기 위해 하는 작업이 특성 추출입니다. 이미지에서는 특성 추출이 도움이 될 수 있습니다.

여기서는 PCA를 이용하여 LFW(Labeled Faced in Wild) 데이터셋의 얼굴 이미지에서 특성을 추출하는 아주 간단한 어플리케이션을 만들어보겠습니다. (200MB 가량)
이 데이터셋은 인터넷에서 내려받은 유명 인사들의 얼굴 이미지들로 2000년 초반 이후의 정치인, 가수, 배우, 운동선수들의 얼굴을 포함하고 있습니다.

여기서는 처리 속도를 높이고자 흑백 이미지를 사용하고 스케일을 줄였습니다.
우선 이미지를 확인하고 시작해봅시다.

``` python
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize= 0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

plt.show()
```
![7](https://i.imgur.com/HWalPCg.png)
LFW 데이터셋에는 62명의 얼굴을 찍은 이미지가 총 3023개가 있으며 크기는 87 * 65 픽셀입니다.

``` python
print("people.images.shape: {}".format(people.images.shape))
print("클래스의 개수: {}".format(len(people.target_names)))
# people.images.shape: (3023, 87, 65)
# 클래스의 개수: 62
```

그렇지만 이 데이터셋은 조금 편중되어 일부 사람으로 편향된 것을 확인할 수 있습니다.
하지만 모든 데이터셋은 편향성이 있으니 이를 위해 사람마다 50개의 이미지만 선택하겠습니다.

``` python
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target== target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people / 255.
```

얼굴 인식이라 하면 통상적으로 새로운 이미지가 데이터베이스에 있는 기존 얼굴 중 하나에 속하는지 찾는 작업입니다.
해결 방법 중 하나는 각 사람을 서로 다른 클래스로 구분하는 분류기를 만드는 것입니다.
하지만 보통 얼굴 데이터베이스에는 사람의 수는 많지만 각 사람에 대한 이미지는 적습니다.

이런 문제 때문에 대부분의 분류기를 훈련시키기 어렵습니다.
그리고 대규모 모델을 다시 훈련시키지 않고도 새로운 사람의 얼굴을 쉽게 추가할 수 있어야 합니다.

간단한 방법으로 분류하려는 얼굴과 가장 비슷한 얼굴 이미지를 찾는 1-최근접 이웃 분류기를 사용할 수 있습니다.
이 분류기는 원칙적으로 클래스마다 하나의 훈련 샘플을 사용합니다.
`KNeighborsClassifier`를 사용하여 확인해봅시다.

``` python
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("{:2f}".format(knn.score(X_test, y_test)))
# 0.23
# 최근접 이웃수를 늘려도 비슷한 결과를 가집니다.
```

정확도가 23% 정도로 나옵니다.
상식 수준의 모델은 랜덤 선택으로 1/62 = 1.6%인것에 비해서는 15배 정도의 좋은 결과지만, 인식률이 좋은 것은 아닙니다.

여기서 이제 PCA를 활용해봅시다. 이미지의 유사도 측정에서 원본 픽셀 공간에서 거리를 계산하는 것은 매우 나쁜 방법입니다.
노이즈 추가로 원하는 거리를 만들어 낼 수 있으며, 평행이동이나 회전이동에도 약한 측정법이기 때문입니다. (그렇기에 딥러닝으로 얼굴의 패턴을 인식하는 것이 가장 좋은 방법이긴 합니다.)

여기서는 주성분으로 변환후 거리를 계산하여 정확도를 높이는 방법에 대해서 진행합니다.
PCA의 **화이트닝(whitening)** 옵셥을 사용해서 주성분의 스케일이 같아지도록 조정합니다.
이는 화이트닝 옵션 없이 변환한 후에 `StandardScaler`를 적용한 것과 같습니다.

PCA 객체를 훈련 데이터로 학습시켜서 처음 100개의 주성분을 추출합니다. 그런 다음 훈련 데이터와 테스트 데이터를 변환합니다.
그런 다음 훈련 데이터와 테스트 데이터를 변환합니다.

> **NOTE** 2차원 그림에서 100개의 주성분이라 하면 이해가 안될 수 있겠지만, 그림의 각 픽셀은 0부터 255사이의 값을 가지는 하나의 차원입니다. 그렇기에 여기서는 각 그림은 5655차원이라고 생각할 수 있습니다.

``` python
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("X_train_pca.shape: {}".format(X_train_pca.shape))
# X_train_pca.shape: (1547, 100)
```

이제 100개의 주성분에 해당하는 특성을 가집니다. 이제 이 데이터를 사용해 1-최근접 이웃 분류기로 이미지를 분류해보겠습니다.

``` python
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("{:.2f}".format(knn.score(X_test_pca, y_test)))
# 0.31
```

그래도 정확도가 31%로 향상되었습니다. 이를 통해 주성분이 이 데이터에서는 더 잘 표현한다고 판단할 수 있습니다.
이미지 데이터에는 계산한 주성분을 쉽게 시각화할 수 있습니다. 몇 개의 주성분을 확인해보겠습니다.
이렇게 얼굴 이미지에서 구한 주성분을 다시 이미지로 나타낸 것을 특별히 **고유얼굴(eigenface)** 라고 합니다.

``` python
fig, axes = plt.subplots(3,5, figsize=(15,12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("PC {}".format(i+1))
```

![8](https://i.imgur.com/ePxLXEf.png)

이 주성분을 완전하게 이해할 수는 없지만 몇몇 주성분이 잡아낸 얼굴 이미지의 특징을 짐작해볼 수 있습니다.
특징에는 명암, 조명, 대략적 형태 등등의 특징이 있습니다.
위에서 나타낸 이미지는 하나의 주성분이므로, 얼굴은 저 이미지들에 가중치를 곱하여 더한 합임을 알 수 있습니다.
하지만 이 방법도 픽셀 기반이므로 얼굴의 배치나 조명에 따라 같은 사람이라도 다르게 인지할 수 있습니다.

## 3.4.2 비음수 행렬 분해(NMF)

**NMF(nor-negative factorization)** 는 유용한 특성을 뽑아내기 위한 또 다른 비지도 학습 알고리즘입니다.
이 알고리즘은 PCA와 비슷하고 차원 축소에도 사용할 수 있습니다.

하지만 PCA에서는 데이터의 분산이 가장 크고 수직인 성분이라면, NMF에서는 음수가 아닌 성분과 계수 값을 찾습니다.
즉, 주성분과 계수가 모두 0보다 크거나 같아야 합니다. 이 방식은 음수가 아닌 특성을 가진 데이터에만 적용할 수 있습니다.

음수 아닌 가중치 합으로 데이터를 분해하는 기능은 여러 사람의 목소리가 담긴 오디오 트랙이나 여러 악기로 이뤄진 음악처럼 독립된 소스를 추가하여 만들어진 데이터에 특히 유용합니다.
이럴 때 NMF는 섞여 있는 데이터에서 원본 성분을 구분할 수 있습니다. 음수로 된 성분이나 계수가 만드는 상쇄 효과를 이해하기 어려운 PCA보다 대체로 NMF의 주성분이 해석하기 쉽습니다.

얼굴 데이터셋에 NMF를 적용해보기 전에 인위적인 데이터셋으로 만든 예를 먼저 보겠습니다.

### 인위적 데이터에 NMF 적용하기

PCA와 달리 NMF에서는 데이터가 양수인지를 먼저 체크해야합니다.
이 말은 데이터가 원점 (0, 0)에서 상대적으로 어디에 놓여 있는지가 NMF에서는 중요하다는 뜻입니다.
그렇기에 원점에서의 벡터를 추출한 것으로 음수 미포함 성분을 이해할 수 있습니다.

다음은 2차원 예제 데이터에 NMF를 적용한 결과입니다.

![9](https://i.imgur.com/QNDbV0y.png)

왼쪽은 성분이 2개인 NMF로 데이터셋의 모든 포인트를 두 성분으로 표현할 수 있습니다.
데이터를 완벽하게 재구성할 수 있을 만큼 성분이 아주 많다면(특성 개수만큼) 알고리즘은 데이터의 각 특성의 끝에 위치한 포인트를 가리키는 방향을 선택할 것입니다.

하나의 성분만 사용한다면 NMF는 데이터를 가장 잘 표현할 수 있는 평균으로 향하는 성분을 만듭니다. PCA와는 반대로 성분 개수를 줄이면 특정 방향이 제거되는 것뿐만 아니라 전체 성분이 바뀝니다. NMF에서 성분은 특정 방식으로 정렬되어 있지 않아, 모든 성분을 동등하게 취급합니다.

NMF는 무작위로 초기화하기 때문에 난수 생성 초깃값에 따라 결과가 달라집니다. 두 개의 성분으로 모든 데이터를 완벽하게 나타낼 수 있는 이런 간단한 예에서는 난수가 거의 영향을 주지 않습니다. 하지만 복잡한 경우에는 큰 차이를 만들 수도 있습니다.

### 얼굴 이미지에 NMF 적용하기

앞에서 사용한 LFW 데이터셋에 NMF를 적용해보겠습니다.
성분을 15개만 추출해보겠습니다.

``` python
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("component {}".format(i))
```

![10](https://i.imgur.com/RkoTYVz.png)

이 성분들은 모두 양수 값이어서 PCA 성분보다 훨씬 더 얼굴 원형처럼 보입니다.
이 성분들이 특별히 강하게 나타난 이미지들을 살펴보겠습니다.

<figure class="half">
    <img src="https://i.imgur.com/P7anrcc.png">
    <img src="https://i.imgur.com/iaNTCUG.png">
    <figcaption> 각각 성분 3과 7의 계수가 큰 얼굴들.</figcaption>
</figure>

3의 경우에는 오른쪽을 보는 경향이 있고, 7의 경우에는 왼쪽을 보는 경향이 있습니다.
이와 같은 패턴을 추출하는 것은 소리, 유전자 표현, 텍스트 데이터처럼 덧붙이는 구조를 가진 데이터에 적합합니다. 인위적인 데이터셋을 사용한 예를 통해서 이에 관해 자세히 살펴보겠습니다.

다음은 세 개의 서로 다른 입력으로부터 합성된 신호입니다.

``` python
S = mglearn.datasets.make_signals()
plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel("t")
plt.ylabel("signal")
plt.margins(0)
```

![13](https://i.imgur.com/rudLYvx.png)

원본신호는 다음과 같지만 우리는 원본 신호가 아닌 합쳐진 신호만 관찰할 수 있는 상황입니다.
이 신호를 분해하여 원본 신호를 복원해야 합니다.
이 신호를 여러 방법(100개의 장치)으로 관찰할 수 있고, 각 장치는 일련의 측정 데이터를 제공한다고 가정합니다.

``` python
# 원본 데이터를 사용해 100개의 측정 데이터를 만듭니다
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("측정 데이터 형태: {}".format(X.shape))
# 측정 데이터 형태: (2000, 100)
```

NMF를 사용해 세 개의 신호를 복원합니다.

``` python
nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print("복원한 신호 데이터 형태: {}".format(S_.shape))
# 복원한 신호 데이터 형태: (2000, 3)
```

비교를 위하여 PCA도 적용합니다.

``` python
pca = PCA(n_components=3)
H = pca.fit_transform(X)
```

NMF와 PCA로 찾은 신호를 나타내었습니다.

``` python
models = [X, S, S_, H]
names = ['signal',
         'signal - original',
         'NMF - Signal',
         'PCA - Signal']
​
fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5},
                         subplot_kw={'xticks': (), 'yticks': ()})
​
for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')
    ax.margins(0)
```

![14](https://i.imgur.com/Jx5CSha.png)

signal은 관측한 데이터 100개 중 3개만을 표현하고 있습니다.
그래프에서 알 수 있듯이 NMF의 경우에는 복원을 하였지만, PCA는 실패한 하였습니다.
PCA의 경우는 변동의 대부분을 첫번째 주성분을 이용하여 데이터를 복원하였습니다.
이 예에서 NMF 성분의 순서가 원본신호와 같지만, NMF는 순서가 없고 우연의 일치입니다.

PCA나 NMF처럼 데이터 포인트를 일정 개수의 성분을 이용해 가중치 합으로 분해할 수 있는 알고리즘이 많이 있습니다. 다음은 패턴 추출 알고리즘을 검색할 수 있는 키워드입니다.

- 독립 성분 분석(ICA)
- 요인 분석(FA)
- 희소 코딩(딕셔너리 학습)


## 3.4.3 t-SNE를 이용한 매니폴드 학습

데이터를 산점도로 시각화할 수 있다는 이점 때문에 PCA를 먼저 시도하긴 합니다.
하지만 알고리즘 자체가 (회전하고 방향을 제거하는 등 복잡한 이유로) 유용성이 떨어집니다.

**매니폴드 학습(manifold learning)** 알고리즘이라고 하는 시각화 알고리즘은 훨씬 복잡한 매핑을 만들어 더 나은 시각화를 제공합니다. 특별히 t-SNE 알고리즘을 많이 사용합니다.

> t-SNE는 t-Distributed Stochastic Neighbor Embedding의 약자입니다.

매니폴드 학습 알고리즘은 시각화를 목적으로 합니다. 그렇기에 3개 이상의 특성을 뽑는 경우가 거의 없습니다.
t-SNE를 포함한 일부 매니폴드 알고리즘은 훈련 데이터는 새로운 표현으로 변환시키지만 새로운 데이터에는 적용할 수 없습니다.

그렇기에 지도 학습용으로는 사용하지 않고, 데이터 탐색을 위주로 사용합니다.
t-SNE의 아이디어는 데이터 포인트 사이의 거리를 가장 잘 보존하는 2차원 표현을 찾는 것입니다.
먼저 t-SNE는 각 데이터 포인트를 2차원에 무작위로 표현한 후 원본 특성 공간에서 가까운 포인트는 가깝게, 멀리 떨어진 포인트는 멀어지게 만듭니다. t-SNE는 멀리 떨어진 포인트보다 가까이 있는 포인트에 더 많은 비중을 둡니다.

> **NOTE** scikit-learn에서 t-SNE 구현은 쿨백-라이블러 발산 목적 함수를 최적화하기 위해 모멘텀을 적용한 배치 경사 하강법을 사용합니다. 기본값은 `bans-hut`옵션이고, 아는 반스-헛 방법으로 그래디언트 계산 복잡도를 O(N^2)에서 O(NlogN)으로 낮춰줍니다. `exact`옵션은 정확한 값을 구해주지만 오래 걸립니다.

scikit-learn에 있는 손글씨 숫자 데이터셋에 t-SNE 매니폴드 학습을 적용합니다. (MNIST와는 다른 데이터셋입니다. 양이 적고 출처가 다릅니다. 이미지 크기도 8 * 8 입니다.)

``` python
from sklearn.datasets import load_digits
digits = load_digits()
​
fig, axes = plt.subplots(2, 5, figsize=(10, 5),
                         subplot_kw={'xticks':(), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)
```

![15](https://i.imgur.com/66VOcAL.png)

샘플 이미지 자체도 픽셀 수가 적다보니 구분하기 어렵게 생겼습니다.
PCA를 사용해 2차원으로 축소해 시각화하겠습니다. 두 개의 주성분을 이용해 그래프를 그리고 각 샘플에 해당하는 클래스는 숫자로 표현하겠습니다.

(예제만으로도 `matplotlib`의 활용을 많이 배울 수 있어 좋은 책입니다.)

``` python
# PCA 모델을 생성합니다
pca = PCA(n_components=2)
pca.fit(digits.data)
# 처음 두 개의 주성분으로 숫자 데이터를 변환합니다
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
    # 숫자 텍스트를 이용해 산점도를 그립니다
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("1st PC")
plt.ylabel("2st PC")
```

![16](https://i.imgur.com/6lRJV49.png)

일부 숫자의 경우에는 잘 분류되지만, 대부분의 숫자는 중첩됨을 알 수 있습니다.
같은 데이터셋을 tsne를 적용해보겠습니다.

`TSNE`는 새 데이터를 변환하는 기능을 제공하지 않습니다. 그렇기에 `transform` 메서드가 없고, 모델을 만들자마자 데이터를 변환해주는 `fit_transform` 메서드를 사용합니다.

``` python
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
# TSNE에는 transform 메소드가 없으므로 대신 fit_transform을 사용합니다
digits_tsne = tsne.fit_transform(digits.data)
​
plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    # 숫자 텍스트를 이용해 산점도를 그립니다
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
```

![17](https://i.imgur.com/wOCvvT9.png)

1과 9를 제외하고는 거의 제대로 분류된 것을 확인할 수 있습니다.
클래스 레이블 정보를 사용하지 않았으므로 비지도 학습입니다.

t-SNE는 매개변수를 약간 조정해야 하지만 기본값으로도 잘 작동하는 경우가 많습니다.
`perplexity`나 `early_exaggeration`를 변경해볼 수 있지만, 보통 효과는 크지 않습니다.
