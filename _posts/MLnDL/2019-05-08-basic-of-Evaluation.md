---
title : "Basic of Evaluation"
category :
  - ML
tag :
  - data
  - Evaluation
  - basic
  - accuracy
  - confusion_matrix
  - precision
  - recall
  - f1
  - roc auc
sidebar_main : true
author_profile : true
use_math : true
header:
  teaser : https://i.imgur.com/AwkwrNc.jpg
  overlay_image : https://i.imgur.com/AwkwrNc.jpg
published : true
---
머신러닝, 지표부터 차근차근

공부할 때마다 혼돈이 와서 이번 기회에 확실하게 하자라는 마음으로 정리해봤습니다. 글을 쓰거나 강의를 찍으면 저도 정리되서 좋은 것 같습니다.

## 머신러닝 평가 지표

머신러닝에서는 다양한 방법으로 모델의 성능을 평가할 수 있습니다. 이번에 살펴볼 지표는 다음과 같습니다.

- 정확도 (Accuracy)
- 오차행렬 (Confusion Matrix)
- 정밀도 (Precision)
- 재현율 (Recall)
- F1 스코어
- ROC AUC

이 지표들은 이진/다중 분류 모두에 적용되는 지표지만, 특히 이진 분류에서 더욱 중요하게 강조하는 지표입니다. 하나하나씩 살펴보도록 하겠습니다.

## 1. 정확도 (Accuracy)

### 설명

**정확도** 는 **실제 데이터에서 예측 데이터가 얼마나 같은지** 를 판단하는 지표입니다.

**정확도 = 예측 결과가 동일한 데이터 건수 / 전체 예측 데이터 건수**

가장 직관적인 모델 예측 성능을 나타내는 평가 지표입니다. 예를 들면 다음과 같습니다.

> 너 OX문제 몇 개 맞췄어? 10개 중에 8개 맞췄어.

대부분의 사람들이 들었을 때, 어느 정도 수긍할 수 있는 수치입니다.
하지만 데이터 과학을 하시는 분이라면, 통계를 다시 한 번 살펴볼 수 있어야합니다.

제가 다닌 고등학교에서 학교의 여성비율은 133명 중에 7명이 여자였습니다.
이때 이름만 듣고 남여를 맞추는 문제를 한다고 가정할 때, 모두 남자로만 찍어도 95% 정도의 정확도를 가집니다.

머신러닝, 캐글에서 가장 유명한 예시 중 하나인 **타이타닉에서 생존자 예측** 문제만 살펴봐도 비슷합니다.
타이타닉에서 생존자의 대다수는 여성입니다. 이때 머신러닝 알고리즘을 사용하지 않더라도 `여성=생존자`와 같이 코드를 작성하더라도 70%이상의 정확도를 가지는 것을 확인할 수 있습니다.

불균형이 심할수록, 모델의 성능을 판단하기에 좋은 지표는 아닙니다.
이런 한계점을 극복하기 위해 통계 선지자 분들은 많은 지표를 만들었습니다.
우선 지표를 설명하기 위해 필요한 오차행렬을 알아봅시다.

### Code

저는 프로그래밍을 하면서 가장 어려운 부분은 사람들이 사용하는 암묵적으로 사용하는 변수명이었습니다.

> 합은 `sum`, 리턴값은 `ret`과 같이 사용하는 것을 의미합니다.

그래서 사이킷런(Scikit-Learn 라이브러리)을 기준으로 어떤 식으로 변수명을 정하는지 설명하고자 합니다.
제가 읽은 캐글 코드와 책들을 바탕으로 조금 적어봤습니다.

분류기(classifier)는 보통 `clf`라는 명칭으로 객체를 만듭니다.
예를 들면 logistic regression이라면 첫 자와 clf를 합쳐 `lr_clf`와 같이 많이 사용합니다.

- Train 데이터는 `X_train`과 `y_train`으로 입력값과 타겟값을 분류합니다.
- Test 데이터는 `X_test`와 `y_test`로 입력값과 타겟값을 분류합니다.

X는 대문자로 y는 소문자로 씁니다.

- 보통 예측값(prediction)은 `pred`를 사용합니다.

결론적으로 정확도를 측정하는 `accuracy_score()`는 다음과 같이 사용합니다.
사용 전에 `sklearn.metrics`에서 불러오는 것을 잊지맙시다.
같은 수의 정답과 예측값을 비교해 정확도를 측정합니다.

``` python
from sklearn.metrics import accuracy_score
clf = MyClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(y_test, pred)
```

## 2. 오차행렬

### 설명

이진 분류에서 성능 지표로 잘 활용되는 **오차행렬(Confusion Matrix, 혼동행렬)** 은 학습된 분류 모델이 예측을 수행하면서 얼마나 헷갈리고 있는지도 함께 보여주는 지표입니다. 어떠한 유형의 예측 오류가 발생하는지를 확인할 수 있습니다.

OX문제를 생각해보겠습니다. 여러분이 정답을 맞추는 케이스는 다음과 같습니다.

- 정답이 O이고, 여러분의 답안이 O인 케이스
- 정답이 X고, 여러분의 답안이 X인 케이스

그렇다면 여러분이 정답을 틀리는 케이스는 다음과 같습니다.

- 정답이 O이고, 여러분의 답안이 X인 케이스
- 정답이 X고, 여러분의 답안이 O인 케이스

이렇게 이진 분류 문제에서 케이스는 총 4가지가 나옵니다. (클래스가 N개인 다중 분류에서는 N^2개의 케이스가 나옵니다.) 그리고 모든 예측은 반드시 이 4가지 케이스에 포함됩니다.

그렇다면 이를 값이 1일 때를 Positive, 0일 때를 Negative라고 생각하면 다음과 같이 행렬로 표현할 수 있습니다. (음성과 양성은 사용자에 따라 다르게 정의할 수 있습니다.)

보통 중점적으로 찾아야 하는 매우 적은 수의 결괏값에 Positive(1)을 설정하고, 그렇지 않은 경우 Negative(1)을 부여하는 경우가 많습니다. 질병, 사기 등의 케이스에서 병에 걸린 상태, 사기 행위를 1로 설정하고 정상 케이스를 0으로 두는 것입니다.

<figure>
    <img src = "https://i.imgur.com/IUrBHiD.png">
    <figcaption>오차행렬</figcaption>
</figure>

T와 F는 각각 True와 False로 실제 값과 예측 값이 같은지를 비교합니다.
뒤의 N, P는 음성(Negative), 양성(Positive)를 의미하고, 이는 예측 값 기준입니다.

표와 각각이 의미하는 알파벳이 무엇을 의미하는지 아는 것이 가장 중요합니다.

그렇다면 정확도도 다시 정의할 수 있습니다. 정확도를 식으로 적어보면 다음과 같습니다.

$$Accuracy = \frac{TN+TP}{TN+FP+FN+TP}$$

이제 이 표를 가지고 나머지 지표들도 살펴보겠습니다.

### Code

이번에도 `sklearn.metrics` 에서 함수를 가져옵니다.
`confusion_matrix()`는 다음과 같이 사용할 수 있습니다. 결과에서 나오는 배열의 형태는 위의 그림과 같습니다. 행은 실제 클래스, 열은 예측 클래스를 나타냅니다.

``` python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)
```

133명 중, 여학생이 7명인 케이스에 대해 모두 남자라고 예측한다면 다음과 같은 결과가 나옵니다.

```
array([[126, 0],    
    [7,0]], dtype=int64)
```

## 3. 정밀도와 재현율

### 설명

불균형 데이터셋에서는 **정밀도(Precision)** 와 **재현율(Recall)** 을 많이 사용합니다. 예측 성능에 좀 더 초점을 맞춘 평가 지표입니다. 우선 공식으로 살펴보겠습니다.

$$Precision = \frac{TP}{FP+TP}$$

$$Recall = \frac{TP}{FN+TP}$$

식을 읽어보면 다음과 같습니다.

- 정확도는 Positive로 예측한 값들 중에 실제로 Positive한 값의 비율
- 재현율은 실제 값이 Positive인 값들 중에 예측을 Positive로 한 값의 비율

정확도은 양성 예측도, 재현율은 민감도(Sensitivity), TPR(True Positive Rate)라고도 불립니다.

이 지표들은 특정 업무에 따라 중요한 지표가 될 수 있습니다. 케이스로 살펴봅시다.

### Case Study

재현율의 경우 다음과 같은 상황을 생각해보겠습니다. 암 판단 모델이 있습니다. 가장 최악의 경우는 무엇일까요?

1. 암환자에게 암을 예측한 경우
2. 암환자가 아닌 사람에게 암을 예측한 경우
3. 암환자에게 암이 아니라고 예측한 경우
4. 암환자가 아닌 사람에게 암이 아니라고 예측한 경우

1번과 4번은 제대로 예측하였고 좋은 대처를 할 수 있습니다. 하지만 2, 3번의 경우에는 분명 힘든 상황이 발생할 수 있습니다. 가장 최악은 3번의 경우입니다. 모델의 판단 결과가 생명에 영향을 미칠 수 있는 상황이기 때문입니다.

이런 케이스의 경우, 재현율을 사용합니다. 실제 암인 환자 중에서 암을 예측한 확률을 구하는 것입니다.

그렇다면 정밀도는 어떤 케이스에서 중요할까요. 이번에는 스팸 메일 구분 모델이 있습니다. 가장 최악의 경우는 무엇일까요?

1. 스팸 메일을 스팸으로 예측한 경우
2. 정상 메일을 스팸으로 예측한 경우
3. 스팸 메일을 정상으로 예측한 경우
4. 정상 메일을 정상으로 예측한 경우

이번에도 1, 4는 좋습니다. 3번의 경우는 번거로움 정도가 있을 수 있습니다. 하지만 2번의 경우에는 정상적인 업무가 불가능합니다.

이런 케이스의 경우, 정밀도를 사용합니다. 스팸으로 예측한 확률 중, 실제 스팸인 확률을 구하는 것입니다.

결론적으로 정밀도와 재현율은 모두 TP를 높이는 데 초점을 둡니다. 하지막 재현율은 FN, 정밀도는 FP를 낮추는 데 초점을 둔다는 차이점이 있습니다.

가장 좋은 방향은 재현율과 정밀도 모두 높은 수치를 얻는 것입니다. 하지만 그것이 과연 가능할까요??

### 정밀도/재현율 Trade-off

분류하려는 업무의 특성에 따라 정밀도 또는 재현율을 강조하기 위한 방법 중 하나로 결정 임계값(Threshold)을 조정하는 방법이 있습니다. 하지만 정밀도와 재현율은 상호 보완적인 평가 지표이기에 강제로 하나를 올리면, 하나는 떨어지기 쉽습니다. (마치 오버피팅같은 느낌입니다.)

이런 것을 정밀도/재현율의 트레이드오프라고 부릅니다.

> 머신러닝에서 가장 고민이 되는 부분은 이런 트레이트 오프인 것 같습니다. 오버피팅과 언더피팅, 정밀도와 재현율 등등 중도를 찾기 위해 가장 많은 시간이 소모되는 것 같습니다.

### 정밀도와 재현율의 거짓말

하지만 정밀도와 재현율도 여러분에게 혼란을 줄 수 있습니다. 다음과 같은 케이스를 살펴보겠습니다.

**정밀도가 100%??**

FP를 0으로 만들면 됩니다. 즉, 정확도가 99%인 스팸메일만 스팸으로 예측하면 됩니다.
그렇다면 식에 따라 정밀도는 100%가 됩니다.

**재현율이 100%??**

모든 환자를 암환자라고 예측하면 됩니다. 그렇다면 FN, 즉 암환자가 아니라고 한 케이스가 0이므로 무조건 재현율이 100%가 됩니다.

이렇게 임계치로 여러분을 속이기는 쉽습니다. 그렇다면 이런 극단적인 경우를 방지하기 위해 이 둘을 조합해서 지표를 만들어보겠습니다. 그것이 바로 F1 Score입니다.

넘어가기 전에 코드만 간략하게 소개하고 넘어가겠습니다.

### Code

각각은 `precision_score()`과 `recall_score()`으로 사용할 수 있습니다.
역시 `sklearn.matrics`에서 불러와서 사용합니다.

``` python
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
```

임계 값의 변경은 다음과 같은 과정을 거쳐야합니다.

1. 분류의 결과를 예측 값이 아닌 예측 확률 값을 변환
2. 원하는 임계 값으로 예측 확률 값을 예측 값으로 변환
3. 다시 오차 행렬에서 원하는 지표 사용

각각의 단계는 다음과 같은 함수를 이용하여 할 수 있습니다.

1. `predict_proba()`를 이용하여 예측 확률을 구합니다. 기존과 같은 포맷이지만 확률로 반환합니다.
2. `sklearn.preprocessing`에서 `Binarizer` 객체를 불러와, `fit_transform()`으로 원하는 임계값으로 예측 값을 만듭니다.

코드로 보면 다음과 같습니다.

``` python
from sklearn.preprocessing import Binarizer

pred_proba = clf.predict_proba(X_test)

# predict_proba는 첫 번째 칼럼에는 0인 확률, 두 번째 칼럼에는 1인 확률을 반환합니다.
# 각 행마다 확률의 합은 1이 되겠죠??
# 보다 필요한 값인 1인 확률만 따로 구해서 사용합니다.
pred_proba_1 = pred_proba[:,1].reshape(-1, 1)

# threshold값보다 작으면 0, 크면 1을 반환
binarizer = Binarizer(threshold=1.5).fit(pred_proba_1)
custom_predict = binarizer.fit_transform(pred_proba_1)

# 원하는 지표 사용
accuracy = accuracy_score(y_test, custom_predict)
precision = precision_score(y_test, custom_predict)
recall = recall_score(y_test, custom_predict)
```

## 4. F1 스코어

### 설명

F1 Score는 정밀도와 재현율을 결합한 지표입니다. 한 쪽으로 치우치지 않는 수치를 나타낼 때 상대적으로 높은 값을 가집니다. F1 스코어의 공식은 다음과 같습니다.

$$F1 = \frac{2}{\frac{1}{recall} + \frac{1}{precision}} = 2 \times \frac{precision \cdot recall}{precision + recall}$$

정밀도와 재현율의 조화평균 입니다. F1 스코어의 최댓값이 1인 것도 쉽게 알 수 있습니다.

### Code

`sklearn.metrics`의 `f1_score`함수를 사용하면 됩니다.

``` python
from sklearn.metrics import f1_score
f1 = f1_score(y_test, pred)
```

## 5. ROC 곡선과 AUC

### 설명

이제 마지막으로 ROC곡선과 이에 기반한 AUC 스코어를 살펴보겠습니다.

**ROC곡선** 은 **Receiver Operation Characteristic Curve** 입니다. 2차대전 때 통신 장비 성능 평가를 위해 고안된 수치로, 한국어로는 수신자 판단 곡선이라고 합니다.

의학 분야에서 많이 사용하나, 머신러닝에서 지표로 많이 사용됩니다.
ROC 곡선은 FPR(X축)과 TPR(Y축)의 관계를 그린 곡선입니다.

- FPR : False Positive Rate
- TPR : True Positive Rate (재현율)

TPR은 이전에 봤던 재현율(민감도)입니다. 이와 대응하는 지표로 TNR이 있습니다. 이는 특이성(Specificity)라고 불립니다.

특이성은 실제값 Negative가 정확히 예측돼야 하는 수준을 의미합니다. 수식으로 쓰면 다음과 같습니다.

$$TNR = \frac{TN}{FP+TN}$$

그리고 이제 ROC에 사용할 FPR은 다음과 같습니다.

$$FPR = \frac{FP}{FP+TN} = 1 - TNR$$

ROC 곡선의 예시를 보고 더 이야기해보겠습니다.

<figure>
    <img src = "https://images.slideplayer.com/32/9873965/slides/slide_18.jpg">
    <figcaption>Example of ROC-AUC </figcaption>
</figure>

가운데 직선은 일반적인 수준(동전 던지기: 확률 50% 랜덤)의 분류 에서 ROC 곡선입니다. ROC 곡선이 직선에 가까울 수록 성능이 떨어지는 것이고, 멀어질수록 성능이 뛰어난 것입니다. 각각은 임계값을 변경하여 다음과 같은 그래프를 만듭니다.

결론적으로 아래에 있는 면적 AUC(Area Under Curve)가 클수록 좋은 값입니다. 대부분의 분류 문제는 최소 0.5 값을 가집니다. 최대는 1로 사각형 전체일때 가능합니다.

### Code

`sklearn.metrics`의 `roc_auc_score`함수를 사용하면 됩니다.

``` python
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, pred)
```

## Reference

- 파이썬 머신러닝 완벽 가이드, 권철민 저, 위키북스
