---
title : \[ML with Python\] 2장. 지도 학습 - 분류와 회귀
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
published : true
---

2.1 분류와 회귀

> 본 문서는 [파이썬 라이브러리를 활용한 머신러닝] 책을 기반으로 하고 있으며, subinium(본인)이 정리하고 추가한 내용입니다. 생략된 부분과 추가된 부분이 있으니 추가/수정하면 좋을 것 같은 부분은 댓글로 이야기해주시면 감사하겠습니다.

1장의 내용은 기초적인 설명이기에 생략하고, 2장부터 시작합니다.
2장의 경우, 내용이 너무 많은 절이 있어 주제 별로 4개로 나누어 포스팅합니다.
(책이 450쪽 정도인데 2장이 110쪽 정도로 거의 25%입니다.)

## 책에서 사용되는 기본 라이브러리

앞으로 사용할 기본적인 모듈만 가볍게 소개합니다.

``` python
from IPython.display import display
import sklearn
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import mglearn # 이 책을 위해 만들어진 라이브러리
import warnings

warnings.simplefilter("ignore") # warning 무시하기
%matplotlib inline
```

대부분 아시겠지만 필요한 라이브러리는 `pip install` 명령어를 이용하여 다운로드하시면 됩니다.
번역서의 `mglearn`의 모듈은 `pip install`로 다운받은 모듈과 살짝 차이가 있다고 합니다.
저는 쉽게 `pip install mglearn`으로 진행하겠습니다.

---

**지도 학습** 은 가장 널리 그리고 성공적으로 사용되는 머신러닝 방법 중 하나입니다.
입력과 출력 데이터가 있고, 주어진 입력으로부터 출력을 예측하고자 할 때 사용합니다.
보통 머신러닝 공부의 시작은 지도 학습이라해도 과언이 아닙니다.

## 2.1.0 INTRO

지도 학습에는 대표적으로 **분류(classification)** 과 **회귀(regression)** 가 있습니다.

### 분류

분류는 미리 정의된, 가능성 있는 여러 **클래스 레이블(class label)** 중 하나를 예측하는 것입니다.
얼굴 인식, 유명한 숫자 판별(MNIST) 등이 이에 속합니다.
분류에는 두 개로만 나누는 **이진 분류(binary classification)** 과 셋 이상의 클래스로 분류하는 **다중 분류(multiclass classification)** 로 나뉩니다.

> 이진 분류에서 한 클래스를 **양성(positive)**, 다른 하나를 **음성(negative)** 클래스라고 합니다. 좋고 나쁘고가 아니기에 분야, 목적에 따라 부를 수 있습니다.

### 회귀

회귀는 연속적인 숫자, 또는 **부동소수점수** (실수)를 예측하는 것입니다. 주식 가격을 예측하여 수익을 내는 알고 트레이딩 등이 이에 속합니다. 분류와 회귀는 연속성이 있는가를 고민하면 쉽게 구분할 수 있습니다.
