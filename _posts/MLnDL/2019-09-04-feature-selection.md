---
title : "Beginner Guide : Feature Selection"
category :
  - ML
tag :
  - feature selection
  - Wrapper
  - Filter
  - Embedded
sidebar_main : true
author_profile : true
use_math : true
header:
  overlay_image : https://images.unsplash.com/photo-1461773518188-b3e86f98242f?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2250&q=80
  overlay_filter: 0.5
published : true
---

이번에는 간단하게(?) Feature Selection을 알아봅시다.

## Feature Selection이란?

Feature Selection은 ML에 있어서 매우 중요한 테크닉이자 기술입니다. 어떻게 보면 여러분의 모델의 성능을 높이기 위해서, 컴페티션에서 좋은 성적을 얻기 위해서는 반드시 필요한 기술 중 하나입니다.
피처를 선택해야하는 것은 어떻게 보면 매우 직관적인 아이디어입니다. raw data에서의 피처를 모두 사용하기에는 computing power과 memory 측면에서 매우 비효율적이기에 일부 필요한 피처만 선택을 한다는 아이디어입니다.

Wikipedia에서는 다음과 같이 Feature Selection을 명시하고 있습니다.

> **feature selection**, also known as variable selection, attribute selection or variable subset selection, is the process of selecting a subset of relevant features (variables, predictors) for use in model construction.

한마디로 `모델을 구성하기 위한 피처(변수)를 선택하는 과정` 입니다. 흔히 많이 이야기하는 Feature Engineering(특성공학)과 Feature Extraction(특성추출)유사합니다. 하지만 분명히 표현 자체는 구분되야 합니다. 간단하게 정리하면 다음과 같습니다. 위키피디아에도 자세히 설명되어 있으니 읽어보면 좋을 것 같습니다.

> FE가 `데이터를 어떻게 유용하게 만들 것인가`의 문제라면 FS는 좀 더 구체적으로 `데이터에서 유용한 피처를 어떻게 선택할 것인가`입니다.

|표현|정의|Wiki|
|-|-|-|
|Feature Engineering | 도메인 지식을 사용하여 데이터에서 피처를 변형/생성 | [wiki](https://en.wikipedia.org/wiki/Feature_engineering) |
|Feature Extraction | 차원축소 등 새로운 중요 피처를 추출 | [wiki](https://en.wikipedia.org/wiki/Feature_extraction) |
|Feature Selection | 기존 피처에서 원하는 피처만 (변경하지 않고) 선택하는 과정 | [wiki](https://en.wikipedia.org/wiki/Feature_selection) |

결론적으로 Feature Selection은 단순하게 하위 세트를 선택하는 과정입니다. 이를 하면 다음과 같은 장점이 있습니다.
(Feature가 적은 데이터에서의 장점을 얻을 수 있다는 것이죠.)

- 사용자가 더 해석하기 쉽게 모델을 단순화
- 훈련 시간의 축소
- 차원의 저주 방지
- 오버피팅을 줄여, 좀 더 gerneralization(일반화)

이제 조오금 더 구체적으로 알아보도록 하겠습니다.

## 어떤 방법론을 사용할 것인가?

Feature Selection을 한다는 것은 하위 셋을 만들기 위한 과정이기에 시간과 자원이 충분하다면(사실 무한하다면) `2^N-1`가지 방법을 모두 테스트하여 구하고자하는 score가 높은 subset을 사용하면 됩니다. 이 방법은 현실적으로 무리기 때문에 평가 메트릭에 따라 적합한 방법을 사용하는 것이 좋습니다.

크게 3가지 분류를 하자면 다음과 같습니다.

- Wrapper method
- Filter method
- Embedded method

조금 더 디테일하게 보도록 하겠습니다.


### Wrapper Method : 유용성을 측정한 방법

<br>

![Wrapper Method](https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Feature_selection_Wrapper_Method.png/600px-Feature_selection_Wrapper_Method.png)

**Wrapper method** 는 예측 모델을 사용하여 피처 subset을 계속 테스트합니다. 이 경우 기존 데이터에서 테스트를 진행할 hold-out set을 따로두어야 합니다.(cross validation) 이렇게 subset을 체크하면 어떤 feature가 필요한지 알 수 있습니다. 하지만 이는 Computing Power가 비약적으로 큰 NP 문제이고, 그렇기에 random hill-climbing과 같은 휴리스틱 방법론을 사용합니다.

예시로는 아래와 같습니다.

- recursive feature elimination(RFE) [(paper)](https://link.springer.com/article/10.1023%2FA%3A1012487302797)
	- scikit-learn에 [함수](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)가 있습니다.
	- SVM을 사용하여 재귀적으로 제거하는 방법입니다.
	- 유사한 방법으로 Backward elimination, Forward elimination, Bidirectional elimination이 있습니다.
- sequential feature selection(SFS)
	- mlxtend에 [함수](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/)가 있습니다.
	- 그리디 알고리즘으로 빈 subset에서 피처를 하나씩 추가하는 방법으로 이루어집니다. 최후에 원하는 피처만 남게 됩니다.

- genetic algorithm
- Univariate selection
- Exhaustive
- mRMR(Minimum Redundancy Maximum Relevance)
	- 피처의 중복성을 최소화하여 Relevancy를 최대화하는 방법
	- [Three Effective Feature Selection Strategies](https://medium.com/ai%C2%B3-theory-practice-business/three-effective-feature-selection-strategies-e1f86f331fb1) 글을 보시는 것을 추천합니다.

### Filter Method : 관련성을 찾는 방법

<br>

![Filter method](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Filter_Methode.png/600px-Filter_Methode.png)

Filter Method는 통계적 측정 방법을 사용하여 피처들의 상관관계를 알아냅니다.

하지만 피처간의 상관계수가 반드시 모델에 적합한 피처라고는 할 수 없고, 세트의 조정이 정확하지 않습니다. 대신 계산속도가 빠르고 피처간 상관관계를 알아내는데 적합하기 때문에 Wrapper method를 사용하기 전에 전처리하는데 사용할 수 있습니다.

아래와 같은 방법이 존재합니다.

- information gain
- chi-square test
- fisher score
- correlation coefficient
	- 흔히 correlation을 heatmap으로 표현하여 시각화합니다.
- variance threshold


### Embedded Method : 유용성을 측정하지만 내장 metric을 사용하는 방법

<br>

![Embedded Method](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Feature_selection_Embedded_Method.png/600px-Feature_selection_Embedded_Method.png)

**Embedded method** 는 모델의 정확도에 기여하는 피처를 학습합니다. 좀 더 적은 계수를 가지는 회귀식을 찾는 방향으로 제약조건을 주어 이를 제어합니다. 예시로는 아래와 같습니다.

- [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) : L1-norm을 통해 제약 주는 방법
- [Ridge](https://en.wikipedia.org/wiki/Tikhonov_regularization) : L2-norm을 통해 제약을 주는 방법
- [Elastic Net](https://en.wikipedia.org/wiki/Elastic_net_regularization) : 위 둘을 선형결합한 방법
- [SelectFromModel](https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2)
	- decision tree 기반 알고리즘에서 피처를 뽑아오는 방법입니다. (RandomForest나 LightGBM 등)
	- scikit-learn에 [함수](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)가 있습니다.

## 주의점

- 훈련할 데이터에서 feature를 고른다면, 훈련 데이터에 과적합될 수 있습니다. 그러므로 훈련 데이터, 테스트 데이터를 제외한 데이터에서 선택하는 것이 중요합니다.
- 모든 데이터에서 feature selection을 진행하면, 교차 검증에서 똑같이 선택된 feature를 사용하게 되므로 결과가 편향될 수 있습니다.

## 그 외 보면 좋은 자료

### sklearn.feature_selection: Feature Selection

scikit-learn의 [Documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)을 확인하면 위의 방법에 관한 방법과 언급하지 않은 방법이 있습니다.

### mlxtend

mlxtend는 ML에서 필요한 기능을 담고 있는 라이브러리입니다. feature_selection 부류의 알고리즘이 꽤 있어 한번 보는 것을 추천합니다.

- [Documentation](http://rasbt.github.io/mlxtend/)
- github : [https://github.com/rasbt/mlxtend](https://github.com/rasbt/mlxtend)

### Code와 함께하는 Article

Filter Method와 Wrapper Method를 파이썬에서 사용하는 방법을 쓴 좋은 글이 있어 추천합니다.

- [Applying Filter Methods in Python for Feature Selection](https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/)
- [Applying Wrapper Methods in Python for Feature Selection](https://stackabuse.com/applying-wrapper-methods-in-python-for-feature-selection/)

[Feature selection techniques for classification and Python tips for their application](https://towardsdatascience.com/feature-selection-techniques-for-classification-and-python-tips-for-their-application-10c0ddd7918b)


### Kaggle Kernels

캐글 커널도 일부 좋은 커널이 있어 추천합니다.

- [Topic 6. Feature Engineering and Feature Selection](https://www.kaggle.com/kashnitsky/topic-6-feature-engineering-and-feature-selection)
- [Introduction to Feature Selection](https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection)
	- 이 글 이후로 아래의 파이썬 클래스를 따로 공유한 것 같습니다.

대부분의 커널은 RFE, SelectFromModel을 많이 소개하고 있습니다. 검색하면 많은 커널이 나오니 한번쯤은 봐도 좋을 것 같습니다.

### FeatureSelector : Python Tool

- article : [A Feature Selection Tool for Machine Learning in Python](https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0)
- github : [https://github.com/WillKoehrsen/feature-selector](https://github.com/WillKoehrsen/feature-selector)

Will Koehrsen님이 만든 툴로 Feature Selection 관련 기능을 클래스로 만든 내용입니다.
많이 사용하는 5개의 feature selection 방법을 사용할 수 있습니다. 위의 방법론과 유사하며 수많은 경험에서 나온 팁으로 보입니다. 참고하시면 좋을 것 같습니다.

1. Features with a high percentage of missing values (빈 데이터가 많은 피처)
2. Collinear (highly correlated) features
3. Features with zero importance in a tree-based model
4. Features with low importance
5. Features with a single unique value

### Check List : An Introduction to Variable and Feature Selection

2003년 An Introduction to Variable and Feature Selection이라는 [paper](http://jmlr.csail.mit.edu/papers/volume3/guyon03a/guyon03a.pdf)에 feature selection에 관한 체크리스트가 있어서 한번 가져왔습니다. (인터랙티브한 그래프로 그리면 이쁠 것 같네요.)

1. **Do you have domain knowledge?** If yes, construct a better set of ad hoc”” features
2. **Are your features commensurate?** If no, consider normalizing them.
3. **Do you suspect interdependence of features?** If yes, expand your feature set by constructing conjunctive features or products of features, as much as your computer resources allow you.
4. **Do you need to prune the input variables (e.g. for cost, speed or data understanding reasons)?** If no, construct disjunctive features or weighted sums of feature
5. **Do you need to assess features individually (e.g. to understand their influence on the system or because their number is so large that you need to do a first filtering)?** If yes, use a variable ranking method; else, do it anyway to get baseline results.
6. **Do you need a predictor?** If no, stop
7. **Do you suspect your data is “dirty” (has a few meaningless input patterns and/or noisy outputs or wrong class labels)?** If yes, detect the outlier examples using the top ranking variables obtained in step 5 as representation; check and/or discard them.
8. **Do you know what to try first?** If no, use a linear predictor. Use a forward selection method with the “probe” method as a stopping criterion or use the `l0-norm` embedded method for comparison, following the ranking of step 5, construct a sequence of predictors of same nature using increasing subsets of features. Can you match or improve performance with a smaller subset? If yes, try a non-linear predictor with that subset.
9. **Do you have new ideas, time, computational resources, and enough examples?** If yes, compare several feature selection methods, including your new idea, correlation coefficients, backward selection and embedded methods. Use linear and non-linear predictors. Select the best approach with model selection
10. **Do you want a stable solution (to improve performance and/or understanding)?** If yes, subsample your data and redo your analysis for several “bootstrap”.

## 표현에 대한 첨언

- Feature의 경우, 한국어로 특성 등을 많이 사용하지만 의미를 위해 `피처`라고 썼습니다.

## Reference

- [An Introduction to Variable and Feature Selection](http://jmlr.csail.mit.edu/papers/volume3/guyon03a/guyon03a.pdf)
- [Wikipedia(Feature selection)](https://en.wikipedia.org/wiki/Feature_selection)
- @MachineLearningMastery : [An Introduction to Feature Selection](https://machinelearningmastery.com/an-introduction-to-feature-selection/)
- @Will_Koehrsen : [A Feature Selection Tool for Machine Learning in Python](https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0)
	- github repo : [Feature Selector: Simple Feature Selection in Python](https://github.com/WillKoehrsen/feature-selector)
- @Abhini_Shetye [Feature Selection with sklearn and Pandas](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)
- [What is the difference between filter, wrapper, and embedded methods for feature selection?](https://sebastianraschka.com/faq/docs/feature_sele_categories.html)

### More Reading

더 읽어보면 좋을 것 같은 자료입니다.

- @Will_Koehrsen : [Automated Feature Engineering in Python](https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219)
- [Feature Selection for Classification: A Review](https://pdfs.semanticscholar.org/310e/a531640728702fce6c743c1dd680a23d2ef4.pdf)
