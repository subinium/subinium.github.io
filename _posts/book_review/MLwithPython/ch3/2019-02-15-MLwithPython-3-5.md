---
title : \[ML with Python\] 3장 비지도 학습과 데이터 전처리 - 군집
category :
  - ML
tag :
  - python
  - deep-learning
  - AI
  - machine learning
  - 머신러닝
  - 비지도 학습
  - 군집
  - Kmeans
  - DBSCAN
  - clustering
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

3.5 군집

> 본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

## 3.5.0 INTRO

앞서 소개한대로 **군집(clustering)** 은 데이터셋을 클러스터라는 그룹으로 나누는 작업입니다.
한 클러스터 안의 데이터 포인트끼리는 매우 비슷하고 다른 클러스터의 데이터 포인트와는 구분되도록 나누는 것이 목표입니다. 분류와 비슷하게 데이터 포인트의 클러스터를 예측 및 할당합니다.

## 3.5.1 k-평균 군집

**k-평균(k-means)** 군집은 가장 간단하고 널리 사용하는 군집 알고리즘입니다.
이 알고리즘은 데이터의 어떤 영역을 대표하는 **클러스터 중심(cluster center)** 을 찾습니다.
이 알고리즘은 다음과 같이 이뤄집니다.

1. 데이터 포인트를 가장 가까운 클러스터 중심에 할당
2. 클러스터에 할당된 데이터 포인트의 평균으로 클러스터 중심을 다시 지정
3. 클러스터에 할당되는 데이터 포인트에 변화가 없을 때 알고리즘 종료.

다음은 예시용 데이터셋에 적용한 예입니다.

![18](https://i.imgur.com/4ONh8XY.png)

그림에서 삼각형은 클러스터의 중심이고, 원은 데이터 포인트입니다.
클러스터는 색으로 구분하였습니다. 본 알고리즘은 3개의 클러스터를 찾도록 지정했습니다.
위의 알고리즘 순서도와 같이 무작위 선정 후 반복을 통해 클러스터를 형성하는 것을 알 수 있습니다.

새로운 데이터 포인트가 주어지면  k-평균 알고리즘은 가장 가까운 클러스터 중심을 할당합니다.
다음 예는 위에서 학습시킨 클러스터 중심의 경계입니다.

![19](https://i.imgur.com/kojcGn9.png)

유클리디안 거리로 보통 거리를 측정하기에 중심들의 외심이라고 예측할 수 있습니다.
scikit-learn에서 k-평균 알고리즘을 사용하는 것도 다른 알고리즘과 비슷합니다.
`KMeans`의 객체를 생성하고 찾고자 하는 클러스터 수를 지정합니다. 그런 다음 `fit` 메서드를 호출합니다.

``` python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(random_state=1)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
```

알고리즘을 적용하면 X에 담긴 각 훈련 데이터 포인트에 레이블이 할당됩니다.

> **NOTE**  `n_clusters`의 기본값은 8이며 잘 사용하지 않습니다. KMeans의 클러스터 레이블 할당은 랜덤입니다.

`kmeans.labels_` 속성에서 이 레이블을 확인할 수 있습니다.

``` python
print("클러스터 레이블 확인 : \n {}".format(kmeans.labels_))
# 클러스터 레이블 확인 :
# [1 0 0 0 2 2 2 0 1 1 0 0 2 1 2 2 2 1 0 0 2 0 2 1 0 2 2 1 1 2 1 1 2 1 0 2 0
# 0 0 2 2 0 1 0 0 2 1 1 1 1 0 2 2 2 1 2 0 0 1 1 0 2 2 0 0 2 1 2 1 0 0 0 2 1
# 1 0 2 2 1 0 1 0 0 2 1 1 1 1 0 1 2 1 1 0 0 2 2 1 2 1]
```

세 개의 클러스터를 지정했으므로 각 클러스터는 0에서 2까지의 번호가 붙습니다.
또 `predict` 메서드를 사용해 새로운 데이터의 클러스터 레이블을 예측할 수 있습니다.
예측은 각 포인트에 가장 가까운 클러스터 중심을 할당하는 것이며 기존 모델은 변경하지 않습니다.
훈련 세트에 대해 `predict` 메서드를 실행하면 `labels_`와 같은 결과를 얻습니다.

``` python
print(np.array_equal(kmeans.labels_,kmeans.predict(X)))
# True
```

군집은 각 데이터 포인트가 레이블을 가진다는 면에서 분류와 비슷하지만, 레이블의 의미를 모르기에 구체적인 정보는 개별 설정해야합니다.
즉 클러스터간 데이터 포인트가 유사하다는 것 이외 정보는 확인을 통해서야 알 수 있습니다.

### k-평균 알고리즘이 실패하는 경우

데이터셋의 클러스터 개수를 정확하게 알고 있더라도 k-평균 알고리즘이 항상 이를 구분해낼수 있는 것은 아닙니다.
각 클러스터를 정의하는 것은 중심 하나뿐이므로 클러스터는 둥근 형태로 나타납니다.
또한 모든 클러스터의 반경이 똑같다고 가정하여 클러스터 중심 사이의 정확히 중간에 경계를 그립니다.
이는 다음과 같은 예상치 않은 결과를 만듭니다.

``` python
X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5], random_state=170)
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)
mglearn.discrete_scatter(X_varied[:,0], X_varied[:,1], y_pred)
plt.legend(["Cluster 1", "Cluster 2", "Cluster 3"], loc='best')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
```

![20](https://i.imgur.com/ppPa172.png)

클러스터의 밀도가 다른 상황에서의 클러스터 할당에 대한 케이스입니다.
몇 포인트에 대해서 cluster 3으로 보이는 포인트들이 애매하게 분류되어 있음을 알 수 있습니다.
k-평균은 또 클러스터에서 모든 방향이 똑같이 중요하다고 가정합니다. 다음으로 볼 데이터는 대각선으로 늘어서 있는 데이터셋입니다.

``` python
# 무작위로 클러스터 데이터를 생성합니다.
X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)

# 데이터가 길게 늘어지도록 변경합니다.
transformation = rng.normal(size=(2,2))
X = np.dot(X, transformation)

# 세 개의 클러스터로 데이터에 KMeans 알고리즘을 적용합니다.
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

# 클러스터 할당과 클러스터 중심을 나타냅니다.
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(
    kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], [0, 1, 2],
    markers='^', markeredgewidth=2)
plt.xlabel("feature 1")
plt.ylabel("feature 2")
```

![21](https://i.imgur.com/9OW2U1i.png)

클러스터가 2장에서 본 two_moons 데이터처럼 더 복잡한 형태라면 k-평균의 성능이 더 나빠집니다.

``` python
# two_moons 데이터를 생성합니다(이번에는 노이즈를 조금만 넣습니다)
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# 두 개의 클러스터로 데이터에 KMeans 알고리즘을 적용합니다
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

# 클러스터 할당과 클러스터 중심을 표시합니다
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm2, s=60, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='^', c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidth=2, edgecolors='k')
plt.xlabel("feature 0")
plt.ylabel("feature 1")
```

![22](https://i.imgur.com/VVETXVD.png)

달 모양 두 개로 나누면 좋겠지만, 이 예시는 k-means의 한계를 잘 보여주고 있습니다.

### 벡터 양자화 또는 분해 메서드로서의 k-평균

k-평균은 군집 알고리즘이지만 PCA나 NMF 같은 분해 알고리즘과 유사한 점을 가지고 있습니다.
PCA는 데이터에서 분산이 가장 큰 방향을 찾으려 하고, NMF는 데이터의 극단 또는 일부분에 상응되는 중첩할 수 있는 성분을 찾습니다.
두 방법 모두 데이터 포인트를 성분의 합으로 나타냅니다.

반면에 k-평균은 클러스터 중심으로 각 데이터 포인트를 표현합니다. 이를 각 데이터 포인트가 클러스터 중심, 즉 하나의 성분으로 표현된다고 볼 수 있습니다. k-평균을 이렇게 각 포인트가 하나의 성분으로 분해되는 관점으로 보는 것을 **벡터 양자화(vector quantization)** 라고 합니다.

PCA, NMF, k-평균에서 추출한 성분과 100개의 성분으로 테스트 세트의 얼굴을 재구성한 것을 나란히 비교해봅시다.
k-means의 경우 재구성은 훈련 세트에서 찾은 가장 가까운 클러스터의 중심입니다.

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=42)
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)
kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)

X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)

fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("extracted feature")
for ax, comp_kmeans, comp_pca, comp_nmf in zip(
        axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
    ax[0].imshow(comp_kmeans.reshape(image_shape))
    ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
    ax[2].imshow(comp_nmf.reshape(image_shape))

axes[0, 0].set_ylabel("kmeans")
axes[1, 0].set_ylabel("pca")
axes[2, 0].set_ylabel("nmf")

fig, axes = plt.subplots(4, 5, subplot_kw={'xticks': (), 'yticks': ()},
                         figsize=(8, 8))
fig.suptitle("reconstructed")
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(
        axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca,
        X_reconstructed_nmf):

    ax[0].imshow(orig.reshape(image_shape))
    ax[1].imshow(rec_kmeans.reshape(image_shape))
    ax[2].imshow(rec_pca.reshape(image_shape))
    ax[3].imshow(rec_nmf.reshape(image_shape))

axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("kmeans")
axes[2, 0].set_ylabel("pca")
axes[3, 0].set_ylabel("nmf")
```

<figure class="half">
    <img src="https://i.imgur.com/HgFyje4.png">
    <img src="https://i.imgur.com/ayvg9Ys.png">
    <figcaption> 성분 100개를 사용한 k-평균, PCA, NMF의 이미지 재구성 비교</figcaption>
</figure>

k-평균을 사용한 벡터 양자화의 흥미로운 면은 데이터 차원보다 더 많은 클러스터를 사용해 데이터를 인코딩할 수 있다는 점입니다. two_moons에서 다시 사용해봅시다.

PCA와 NMF를 이용해 차원 축소를 하면 데이터 구조가 파괴되므로 사용할 수 없습니다. 하지만 많은 클러스터 중심을 사용한 k-평균은 데이터를 더 잘 표현할 수 있습니다.

``` python
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired', edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60,
            marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired', edgecolors='black')
plt.xlabel("feature 0")
plt.ylabel("feature 1")
```

![25](https://i.imgur.com/qcL7o58.png)

10개의 클러스터로 나누었습니다. 이런 구조라면 2개의 반달 모양을 구분할 수 있을 것 같습니다.
이 데이터는 kmeans 모델의 `transform` 메서드를 사용하여 구할 수 있습니다.

> **NOTE**  `transform` 메서드가 반환하는 값은 데이터 포인트에서 각 클러스트 중심까지의 거리이므로 그 크기는 '샘플 개수 * 클러스터 수'입니다.

k-평균은 비교적 이해하기 쉽고 구현도 쉬울 뿐만 아니라 비교적 빠르게 때문에 가장 인기 있는 군집 알고리즘입니다.
대용량 데이터셋에도 잘 작동하고, scikit-learn은 아주 큰 대규모 데이터셋을 처리할 수 있는 MiniBatchMeans도 제공합니다.

> **NOTE** MiniBatchMeans는 알고리즘이 반복될 때 전체 데이터에서 일부를 무작위로 선택해 클러스터의 중심을 계산합니다. batch_size 매개변수로 미니 배치 크기를 지정할 수 있고 기본값은 100입니다.

k-평균의 단점 하나는 무작위 초기화를 사용하여 알고리즘의 출력이 난수 초깃값에 따라 달라진다는 점입니다.
기본적으로 scikit-learn에서는 서로 다른 난수 초깃값으로 10번 반복하여 최선의 결과를 만듭니다. (클러스터 분산의 합이 작은 것)
또 다른 단점으로는 클러스터의 모양을 가정하고 있어서 활용 범위가 비교적 제한적이며, 또 찾으려 하는 클러스터의 개수를 지정해야만 한다는 것입니다. 이런 단점을 개선한 두 가지 군집 알고리즘을 더 살펴보겠습니다.

## 3.5.2 병합 군집

**병합 군집(agglomerative clustering)** 은 다음과 같은 원리로 만들어진 군집 알고리즘의 모음을 말합니다.

- 시작할 때 각 포인트를 하나의 클러스터로 지정
- 특정 종료 조건을 만족할 때까지 가장 비슷한 두 클러스터를 합칩니다.

scikit-learn에서 사용하는 종료 조건은 클러스터 개수로, 지정된 개수의 클러스터가 남을 때까지 비슷한 클러스터를 합칩니다.
`linkage` 옵션은 가장 비슷한 클러스터를 측정하는 방법을 지정합니다. 항상 두 클러스터 사이에서만 측정이 이뤄집니다.
다음은 scikit-learn에 구현된 옵션입니다.

- `ward` : 기본값인 ward 연결. 분산을 가장 작게 증가시키는 두 클러스터를 합칩니다. 비교적 비슷한 크기의 클러스터가 만들어집니다.
- `average` : average 연결은 클러스터 포인트 사이의 평균 거리가 가장 짧은 두 클러스터를 합칩니다.
- `complete` : complete 연결은 클러스터 포인트 사이의 최대 거리가 가장 짧은 두 클러스터를 합칩니다.

`ward`가 대부분의 데이터셋에 알맞기 때문에 예제에서 이 옵션을 사용하겠습니다. 클러스터에 속한 포인트 수가 많이 다를 땐 average나 complete가 나을 수 있습니다.

다음은 2차원 데이터셋에서 세 개의 클러스터를 찾기 위한 병합 군집의 과정입니다.

![26](https://i.imgur.com/JWDGZJ6.png)

초기에는 각 포인트가 하나의 클러스터입니다. 그 다음 각 단계에서 가장 가까운 두 클러스터가 합쳐집니다.
네 번째 단계까지는 포인트가 하나뿐인 클러스터 두 개가 선택되어 합쳐져서 두 개의 포인트를 가진 클러스터가 확장되었습니다.
이런 식으로 계속 합쳐지며 클러스터 3개가 남게되며 이때 알고리즘이 멈춥니다.

앞서 사용한 세 개의 클러스터가 있는 데이터셋으로 병합 군집이 작동하는 것을 확인해보겠습니다.
알고리즘 작동 특성상 병합 군집은 새로운 데이터 포인트에 대해서는 예측을 할 수 없습니다.

그러므로 병합 군집은 `predict` 메서드가 없습니다. 그래서 클러스터를 만들고 소속 정보를 얻기 위해 `fit_predict` 메서드를 사용합니다.

``` python
from sklearn.cluster import AgglomerativeClustering
X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc="best")
plt.xlabel("feature 0")
plt.ylabel("feature 1")
```

![27](https://i.imgur.com/UJxjK41.png)

예상대로 알고리즘은 클러스터를 완벽하게 찾았습니다. scikit-learn에서 병합 군집 모델을 사용하려면 클러스터 개수를 지정해야 하지만, 병합 군집이 적절한 개수를 선택하는 데 도움을 주기도 합니다.

### 계층적 군집과 덴드로그램

병합 군집은 **계층적 군집(hierarchical clustering)** 을 만듭니다. 군집이 반복하여 진행되면 모든 포인트는 하나의 포인트를 가진 클러스터에서 시작하여 마지막 클러스터까지 이동하게 됩니다.
각 중간 단계는 데이터에 대한 클러스터를 생성합니다. 이는 가능한 모든 클러스터를 연결해보는데 도움이 됩니다.
다음은 병합 군집 처음 예시의 데이터를 좀 더 디테일하게 표현한 것입니다. 각 클러스터가 더 작은 클러스터로 어떻게 나뉘는지 보여줍니다.

![28](https://i.imgur.com/xeBdJT7.png)

기 그래프는 계층 군집의 모습을 자세히 나타내지만 2차원에서만 가능하고 3차원에서는 사용할 수 없습니다.
하지만 계층 군집을 시각화하는 다른 도구인 **덴드로그램(dendrogram)** 은 다차원 데이터셋을 처리할 수 있습니다.

아쉽게 scikit-learn은 아직까지 덴드로그램을 그리는 기능을 제공하지 않습니다.
하지만 Scipy를 사용해 손쉽게 만들 수 있습니다. Scipy 군집 알고리즘은 scikit-learn의 군집 알고리즘과는 인터페이스가 조금 다릅니다.
Scipy는 데이터 배열 X를 받아 계층 군집의 유사도가 들어있는 연결 배열을 반환하는 함수를 제공합니다.
이 연결 배열을 Scipy의 dendrogram 함수에 넣어 그래프를 그릴 수 있습니다.

``` python
# SciPy에서 ward 군집 함수와 덴드로그램 함수를 임포트합니다
from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)
# 데이터 배열 X 에 ward 함수를 적용합니다
# SciPy의 ward 함수는 병합 군집을 수행할 때 생성된
# 거리 정보가 담긴 배열을 리턴합니다
linkage_array = ward(X)
# 클러스터 간의 거리 정보가 담긴 linkage_array를 사용해 덴드로그램을 그립니다
dendrogram(linkage_array)

# 두 개와 세 개의 클러스터를 구분하는 커트라인을 표시합니다
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' 2 clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' 3 clusters', va='center', fontdict={'size': 15})
plt.xlabel("sample number")
plt.ylabel("cluster distance")
```

![29](https://i.imgur.com/iS08kD0.png)

덴드로그램에서 데이터 포인트는 맨 아래 나타납니다. 이 포인트들은 leaf로 하는 트리가 만들어지며 새로운 부모 노드는 두 클러스터가 합쳐질 때 추가됩니다.

높이 차이를 통해 만들어진 순서를 확인할 수 있습니다. y축은 또한 클러스터간 거리를 나타냅니다. 이 덴드로그램에서는 루트노드에서 시작된 3개의 가장 긴 파란 실선이 있으니 클러스터가 3개로 표현할 수 있음을 알 수 있습니다.

하지만 이 알고리즘 역시 two_moons 데이터셋과 같은 복잡한 형상을 구분하지 못합니다. 다음 알고리즘을 통해 가능할지 알아봅시다.

## 3.5.3 DBSCAN

**DBSCAN(density-based spatial clustering of applications with noise)** 은 아주 유용한 군집 알고리즘입니다.
번역하면 밀도 기반 클러스터링이라고 할 수 있습니다.

이 알고리즘의 주요 장점은 클러스터의 개수를 미리 지정할 필요가 없다는 점입니다.
이 알고리즘은 복잡한 형상도 찾을 수 있으며, 어떤 클래스에도 속하지 않는 포인트를 구분할 수 있습니다.
DBSCAN은 이전의 군집 알고리즘보다는 느리지만 비교적 큰 데이터셋에도 적용할 수 있습니다.

DBSCAN은 특성 공간에서 가까이 있는 데이터의 밀도가 높은 곳의 포인트를 찾습니다. (**밀집 지역(dense region)** 이라는 특성 공간)
DBSCAN의 아이디어는 데이터의 밀집 지역이 한 클러스터를 구성하며 비교적 비어있는 지역을 경계로 다른 클러스터와 구분된다는 것입니다.

밀집 지역에 있는 포인트를 **핵심 샘플** 이라고 하며 다음과 같이 정의합니다.

- 두 개의 매개변수 `min_samples`와 `eps`가 있습니다.
- `eps` 거리(기본값 euclidean) 안에 데이터가 `min_samples` 개수만큼 들어 있으면 이 데이터 포인트를 핵심 샘플로 분류
- `eps`보다 가까운 핵심 샘플은 동일한 클러스터로 합쳐집니다.

이 알고리즘은 다음과 같은 방식으로 진행됩니다.

1. 무작위로 포인트 선택
2. 선택 포인트에 대해 eps 거리 안의 모든 포인트 탐색
    - `eps` 거리 안에 있는 포인트 수가 `min_samples`보다 적다면 어떤 클래스에도 속하지 않는 `noise`로 분류
    - `eps` 거리 안에 있는 포인트 수가 `min_samples`보다 많다면 핵심 샘플로 레이블하고 새로운 클러스터 레이블 할당
3. 할당된 포인트의 모든 이웃 확인
    - 클러스터에 할당되지 않았다면 바로 전에 만든 클러스터 레이블을 할당
    - 만약 핵심 샘플이면 그 포인트의 이웃을 차례로 방문
4. 3과정을 반복하여 `eps` 거리 안에 더 이상 핵심 샘플이 없을 때 까지 반복후, 방문 못한 데이터에 대해 1부터 반복

위의 알고리즘에서 포인트(샘플)의 종류는 3가지로 설명할 수 있습니다. **핵심 포인트, 경계 포인트, 잡음 포인트** 입니다.
경계 포인트는 핵심 포인트에서 eps거리 안에 있는 포인트를 의미합니다.

위 알고리즘을 한 데이터셋에 여러 번 실행하면 핵심 포인트의 군집은 항상 같고 매번 같은 포인트를 잡음으로 레이블합니다. 그러나 경계 포인트는 한 개 이상의 클러스터 핵심 샘플의 이웃일 수 있습니다. 보통 경계 포인트는 많지 않으며 포인트 순서 때문에 받는 영향도 적어 중요한 이슈는 아닙니다.

이제 병합 군집에 사용했던 데이터셋에 DBSCAN을 적용해보겠습니다. 병합 군집과 마찬가지로 새로운 테스트 데이터에 대해 예측할 수 없으므로 `fit_predict` 메서드를 사용하여 군집과 클러스트 레이블을 한 번에 계산합니다.

``` python
from sklearn.cluster import DBSCAN
X, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print("클러스터 레이블:\n{}".format(clusters))
# 클러스터 레이블:
# [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
# 모든 포인트에 잡음 포이인트를 의미하는 -1이 할당되었고, 이는 작은 샘플에 적합하지 않은 eps와 min_samples 기본값 때문입니다.
# 여러가지 변수 값에 대한 클러스터 할당을 체크해봅시다.
mglearn.plots.plot_dbscan()
# min_samples: 2 eps: 1.000000  cluster: [-1  0  0 -1  0 -1  1  1  0  1 -1 -1]
# min_samples: 2 eps: 1.500000  cluster: [0 1 1 1 1 0 2 2 1 2 2 0]
# min_samples: 2 eps: 2.000000  cluster: [0 1 1 1 1 0 0 0 1 0 0 0]
# min_samples: 2 eps: 3.000000  cluster: [0 0 0 0 0 0 0 0 0 0 0 0]
# min_samples: 3 eps: 1.000000  cluster: [-1  0  0 -1  0 -1  1  1  0  1 -1 -1]
# min_samples: 3 eps: 1.500000  cluster: [0 1 1 1 1 0 2 2 1 2 2 0]
# min_samples: 3 eps: 2.000000  cluster: [0 1 1 1 1 0 0 0 1 0 0 0]
# min_samples: 3 eps: 3.000000  cluster: [0 0 0 0 0 0 0 0 0 0 0 0]
# min_samples: 5 eps: 1.000000  cluster: [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
# min_samples: 5 eps: 1.500000  cluster: [-1  0  0  0  0 -1 -1 -1  0 -1 -1 -1]
# min_samples: 5 eps: 2.000000  cluster: [-1  0  0  0  0 -1 -1 -1  0 -1 -1 -1]
# min_samples: 5 eps: 3.000000  cluster: [0 0 0 0 0 0 0 0 0 0 0 0]
```
![30](https://i.imgur.com/121ds7J.png)

이 그래프에서 클러스터에 속한 포인트는 색을 칠하고 잡음 포인트는 하얀색으로 남겨뒀습니다.
핵심 샘플은 크게 표시하고 경계 포인트는 작게 나타냈습니다.
`eps`를 증가시키면 하나의 클러스터에 더 많은 포인트가 포함됩니다.
`eps`를 너무 크게하면 클러스터가 단 하나의 클러스터에 속하고, 너무 작으면 어떤 포인트도 핵심이 안되어 잡음이 많아집니다.
`min_samples`를 증가시키면 핵심 포인트 수가 줄어들며 잡음 포인트가 늘어납니다.

DBSCAN은 클러스터의 개수를 지정할 필요가 없지만 매개변수를 통해 제어합니다. 적절한 eps 값을 쉽게 찾으려면 StandardScaler나 MinMaxScaler로 모든 특성의 스케일을 비슷한 범위로 조정해주는 것이 좋습니다.
아래는 two_moons 데이터셋에 DBSCAN을 적용한 결과입니다. 기본 설정값으로 진행하여 두 개의 반달을 정확히 찾아 구분하였습니다.

 ``` python
 X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

 # 평균이 0, 분산이 1이 되도록 데이터의 스케일을 조정합니다
 scaler = StandardScaler()
 scaler.fit(X)
 X_scaled = scaler.transform(X)

 dbscan = DBSCAN()
 clusters = dbscan.fit_predict(X_scaled)
 # 클러스터 할당을 표시합니다
 plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60, edgecolors='black')
 plt.xlabel("feature 0")
 plt.ylabel("feature 1")
 ```

DBSCAN에서 주의할 점은 클러스터 할당값을 이용해 다른 배열의 index로 사용하면 안된다는 것입니다. -1 값을 가지는 잡음 포인트가 어떤 결과를 만들지 모르기 때문입니다.

## 3.5.4 군집 알고리즘의 비교와 평가

군집 알고리즘을 적용하는 데 어려운 점은 하나의 알고리즘이 잘 작동하는지 평가하거나 여러 알고리즘의 출력을 비교하기 매우 어렵다는 것입니다.
지금까지의 알고리즘을 비교해봅시다.

### 타깃값으로 군집 평가하기

군집 알고리즘의 결과를 실제 정답 클러스터와 비교하여 평가할 수 있는 지표들이 있습니다. 1(최적일 때)과 0(무작위 분류일 때) 사이의 값을 제공하는 **ARI(adjusted rand index)** 와 **NMI(normalized mutual information)** 가 가장 널리 사용하는 지표입니다. (ARI는 음수가 될 수 있습니다.)

다음은 ARI를 사용해서 알고리즘을 비교합니다. 두 클러스터에 무작위로 포인트를 할당해 함께 비교했습니다.

``` python
from sklearn.metrics.cluster import adjusted_rand_score
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# 평균이 0, 분산이 1이 되도록 데이터의 스케일을 조정합니다
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3),
                         subplot_kw={'xticks': (), 'yticks': ()})

# 사용할 알고리즘 모델을 리스트로 만듭니다
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
              DBSCAN()]

# 비교를 위해 무작위로 클러스터 할당을 합니다
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

# 무작위 할당한 클러스터를 그립니다
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
                cmap=mglearn.cm3, s=60, edgecolors='black')
axes[0].set_title("Randomly - ARI: {:.2f}".format(adjusted_rand_score(y, random_clusters)))

for ax, algorithm in zip(axes[1:], algorithms):
    # 클러스터 할당과 클러스터 중심을 그립니다
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters,
               cmap=mglearn.cm3, s=60, edgecolors='black')
    ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__,  adjusted_rand_score(y, clusters)))
```

![31](https://i.imgur.com/kLJNHDV.png)

클러스터를 무작위로 할당했을 때의 ARI의 점수는 0이고, DBSCAN은 점수가 1입니다.

군집 모델을 평가할 때는 ARI나 NMI 등 군집용 측정 도구를 사용해야하며, `accuracy_score` 를 사용해서는 안됩니다. (하나하나 비교)
군집은 순서가 중요한 것이 아니고, 같은 클러스터에 같은 포인트가 들어있는 지 중요하기 때문입니다.

### 타깃값 없이 군집 평가하기

ARI의 문제점은 타깃값을 알고 있어야한다는 것입니다. 타깃값이 있으면 지도 학습 모델을 만들기에 비지도 학습 모델에는 적합하지 않습니다.
그렇기에 ARI나 NMI의 경우에는 애플리케이션의 성능 평가가 아니라 알고리즘을 개발할 때나 도움이 됩니다.

타깃값이 필요 없는 군집용 지표로 **실루엣 계수(silhouette coefficient)** 가 있습니다. 그러나 이 지표는 실제로 잘 작동하지 않습니다.
실루엣 점수는 클러스터의 밀집 정도를 계산하는 것으로, 높을수록 좋으며 최대 점수는 1입니다.
밀집된 클러스터가 좋긴 하지만 모양이 복잡할 때는 평가가 잘 들어맞지 않습니다. (원형 클러스터에서 값이 더 높게 나옴)

다음은 two_moons 데이터셋에서 실루엣 점수를 사용해 결과를 비교한 예입니다.

``` python
from sklearn.metrics.cluster import silhouette_score

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# 평균이 0, 분산이 1이 되도록 데이터의 스케일을 조정합니다
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})

# 비교를 위해 무작위로 클러스터 할당을 합니다
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

# 무작위 할당한 클러스터를 그립니다
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
                cmap=mglearn.cm3, s=60, edgecolors='black')
axes[0].set_title("Randomly : {:.2f}".format(
        silhouette_score(X_scaled, random_clusters)))

algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
              DBSCAN()]

for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    # 클러스터 할당과 클러스터 중심을 그립니다
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3,
               s=60, edgecolors='black')
    ax.set_title("{} : {:.2f}".format(algorithm.__class__.__name__, silhouette_score(X_scaled, clusters)))
```

![32](https://i.imgur.com/Etw7WFc.png)

결과에서 볼 수 있듯이 k-평균이 DBSCAN보다 실루엣 점수가 높은 것을 확인할 수 있습니다.

클러스터 평가에 더 적합한 전략은 견고성 기반의 지표입니다.
데이터 잡음 포인트를 추가하거나 여러 가지 매개변수 설정으로 알고리즘을 실행하고 그 결과를 비교하는 것입니다.
매개변수와 데이터 변화를 주며 반복해도 결과가 일정하다면 신뢰할만 하다고 할 수 있지만, 아직 scikit-learn에는 기능이 구현되어 있지 않습니다.

결국 클러스터가 우리 기대에 부합하는지 알 수 있는 방법은 클러스터를 직접 확인하는 것뿐입니다.

### 얼굴 데이터셋으로 군집 알고리즘 비교

**생략**

## 3.5.5 군집 알고리즘 요약

군집 알고리즘을 적용하고 평가하는 것이 매우 정성적인 분석 과정이며 탐색적 데이터 분석 단계에 크게 도움될 수 있다는 것을 알 수 있었습니다.
세 가지 알고리즘을 보았고, 각각 군집을 세밀하게 조절할 수 있는 방법을 배웠습니다.

- k-평균, 병합 군집 : 개수
- DBSCAN : eps, min_samples

각 알고리즘은 조금씩 다른 장점을 가지고 있습니다.

- k-평균 : 클러스터 중심을 이용함. 벡터 양자화
- DBSCAN : 잡음 포인트 인식, 개수 자동 지정, 복잡한 모양 가능
- 병합 군집 : 전체 데이터의 분할 게층도를 만들어주며 덴드로그램을 사용해 손쉽게 확인 가능
