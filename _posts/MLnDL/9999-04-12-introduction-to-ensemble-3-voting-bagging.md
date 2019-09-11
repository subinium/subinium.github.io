---
title : "Introduction to Ensemble Learning : Part 2"
category :
  - ML
tag :
  - machine learning
  - Ensemble
  - basic
  - Boosting
  - Stacking
sidebar_main : true
author_profile : true
use_math : true
header:
  teaser : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTAjdD3BHViwTrrlXcdJ9rdJHWHtqTb3Ba6e9DCuF7rRQZ7pLJT
  overlay_image : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTAjdD3BHViwTrrlXcdJ9rdJHWHtqTb3Ba6e9DCuF7rRQZ7pLJT
published : False
---
Part2. Voting/Averaging & Bagging with Code

## 0. Intro

> 다음 코드들은 [Sklearn 공식 Documentation](https://scikit-learn.org/stable/documentation.html)와 [A Comprehensive Guide to Ensemble Learning (with Python codes)](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/) 포스팅에서 많이 가져왔습니다.

Part 1에 이어 Part 2,3,4는 내용에 대한 심화와 코드 예시입니다.

- [Part 1. What is Ensemble Learning](/introduce-to-ensemble-1)
- Part 2. Voting/Averaging & Bagging with Code
- Part 3. Boosting with Code
- Part 4. Stacking with Code

<figure>
    <img src = "https://i.imgur.com/iN6lUw7.png" >
    <figcaption> 생각보다 공식 문서에 잘 되어있습니다. </figcaption>
</figure>


> 본 포스트의 scikit-learn은 버전 0.20.3을 기준으로 작성하였습니다.

## 1. Voting & Averaging

보팅과 애버리징은 다른 앙상블에 비해 간단하고, 실제로 단순하게 코드로 구현도 간단합니다.
추가로 Sklearn에서는 voting 분류 모델인 VotingClassifier을 제공하고 있습니다.

### 1-1. Hard/Soft Voting

보팅은 각 모델의 약점을 보완할 수 있으며, 비슷한 성능을 내는 모델들을 통합해 성능을 높이는 것을 목표로 합니다. Hard와 Soft는 다음과 같이 정의할 수 있습니다.

- hard voting : 개별 모형의 결과 기준 단순 투표
- soft voting : 개별 모형의 조건부 확률 기준 가중치 투표

코드로 더 알아보도록 하겠습니다. Sklearn 공식 문서에서 가져온 예시입니다.
싸이킷런에서 제공하는 붓꽃 데이터인 iris.data를 사용하여 진행하는 예시입니다.

``` python
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

# 데이터 불러오기
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

# 사용할 비슷한 성능의 예측기
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

# VotingClassifier
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

# cross_val_score로 계산한 분류기들의 정확도
for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

```

``` shell
Accuracy: 0.95 (+/- 0.04) [Logistic Regression]
Accuracy: 0.94 (+/- 0.04) [Random Forest]
Accuracy: 0.91 (+/- 0.04) [naive Bayes]
Accuracy: 0.95 (+/- 0.04) [Ensemble]
```

본 코드에서 보면 `VotingClassifier` 에는 총 2가지 파라미터가 있습니다.

1. `estimators` : 개별 모델의 목록, list 형태로 입력합니다.
2. `voting` : hard 또는 soft로 설정하여 원하는 보팅 방식을 선택합니다. default는 hard입니다.

soft voting에서는 다음과 같이 가중치를 두어 사용할 수도 있습니다.

``` python
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(gamma='scale', kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2, 1, 2])
```

여기서는 새로운 파라미터 `weights`가 나옵니다.

- `weights` : 모델의 가중치 리스트

그리고 GridSearchCV를 활용하여 하이퍼 파라미터를 조정할 수 있습니다.

``` python
from sklearn.model_selection import GridSearchCV
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid = grid.fit(iris.data, iris.target)
```

### 1-2. Simple/Weighted Averaging

애버리징 자체는 따로 구현된 코드가 없기에 직접 구현해야 합니다.

## 2. Bagging

## Reference

세상에는 좋은 자료가 너무 많습니다. 시간이 된다면 아래 링크와 책 모두 읽는 것을 추천합니다. Part 1에서 언급한 Reference를 제외한 Reference 입니다.

- [EnsembleVoteClassifier](http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/#ensemblevoteclassifier)
