---
title : Introduction to Deep Learning Normalization
category :
  - ML
tag :
  - machine learning
  - deep learning
  - normalization
  - batch normalization
  - weight normalization
  - layer normalization
  - instance normalization
  - pros and cons
sidebar_main : true
author_profile : true
use_math : true
header:
  teaser : /assets/images/category/ml.jpg
  overlay_image : /assets/images/category/ml.jpg

---

수 많은 정규화들을 한번 가볍게 읽어봅시다.

> 본 글은 [An Overview of Normalization Methods in Deep Learning](http://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/)을 바탕으로 추가 및 수정을 한 글입니다. 향후 부족한 부분은 추가될 예정입니다.


머신러닝/딥러닝에서 Normalization은 항상 중요한 문제 중에 하나였습니다.

Normalization은 모델의 훈련을 더욱 효과적으로 할 수 있게 만드는 과정입니다. [link](https://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network)

훈련된 모델들이 보다 일반화 성능을 향상시키기 위한 필수적인 과정이라고 할 수 있습니다.  

머신러닝/딥러닝으로 문제를 해결하기 위해서는 적재적소에 normalization 방법을 사용하는 것이 중요합니다.
이번 포스팅에서는 이런 정규화들의 종류를 가볍게 살펴보겠습니다.

> 일부 내용의 경우, 논문과 글들을 요약하는 과정에서 잘못된 부분이 있을 수 있으니 많은 피드백해주시면 감사하겠습니다.

## 1. Normalization vs.Regularization vs Standardization

이 개념은 매우 혼동되기 쉬운 표현입니다.
우선 이 세 단어에 대한 정의를 우선으로 하고 가겠습니다.

### 1.1 Normalization

Normalization은 흔히 한국어로 **정규화**(*앞으로 문서에서는 정규화라는 표현을 사용하도록 하겠습니다.*)라고 불립니다.
머신러닝 또는 딥러닝에서 정규화의 목표는 값 범위의 차이를 왜곡시키지 않고 데이터 세트를 공통 스케일로 변경하는 것입니다.

데이터 처리에 있어 각 특성의 스케일을 조정한다는 의미로, **feature scaling** 이라고도 불립니다.

Normalization는 흔히 스케일 조정 중 **Min-Max Scaler** 의 의미로 사용되기도 합니다. (데이터를 0과 1사이로 스케일 조정하는 방법)

$$\frac{x-x_{min}}{x_{max}-x_{min}}$$

### 1.2 Regularization

Regularization 또한 *정규화* 라고 번역되는 경우가 많지만 ***일반화*** 라고 번역하기도 합니다. 그렇다면 또 Generalization과 의미가 혼동되기도 합니다. 규제라고도 번역하지만 페널티로 모델을 제어하는 방식에는 유효하지만 또 다른 방식에는 어색한 표현입니다.

보통 모델의 설명도를 유지하면서 모델의 복잡도를 줄이는 방식을 말합니다.

- Early stopping
- Noisy input
- drop-out
- Pruning & feature selection
- emsemble

등이 이에 속합니다.



### 1.3 Standardization

Standardization는 표준화로 흔히 **Standard Scaler** 또는 **z-score normalization** 을 의미합니다.
기존 데이터를 평균 0, 표준 편차 1인 표준분포의 꼴 데이터로 만드는 것을 의미합니다.

평균을 기준으로 얼마나 떨어져 있는지를 살펴볼 때 보통 사용합니다. 보통 데이터 분포가 가우시안 분포를 따를 때 유용합니다.

$$\frac{x-\bar{x}}{\sigma}$$

표준화와 정규화는 경우에 따라 다르게 정의되고는 합니다. 그렇기에 때에 따라 의미를 살펴보는 것이 좋습니다.

## 2. Batch Normalization과 그 효율성

### 2.1 Batch Normalization이란

Batch Normalization은 2015년 [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf) 논문에 설명되어 있는 기법입니다.

이 정규화 방법은 Gradient Vanishing / Gradient Exploding이 일어나지 않도록 하는 아이디어 중의 하나입니다.

- **Gradient Vanishing** : 신경망에서 손실 함수의 gradient 값이 0에 근접하며, 훈련을 하기 힘들어지는 문제
- **Gradient Exploding** : 심층 신경망 또는 RNN에서 error gradient가 축적되어, 가중치 업데이트시 overflow되거나 NaN값이 되는 현상

이런 문제들은 원래 Activation 함수를 ReLU로 변환, 초깃값 튜닝, 작은 learning rate 값으로 해결했습니다.
Batch Normalization 방법은 training 과정 자체를 안정화하여 학습 속도를 가속화하기 위해 만들어졌습니다.

논문에서는 위에서 언급된 문제의 근본적인 원인은 **Internal Covariance Shift** 라는 문제라고 주장(경험적으로 그렇다~)하고 있습니다. 이는 네트워크의 각 층이나 활성 함수마다 입력의 값들의 분포가 계속 바뀌는 현상을 의미합니다.

그래서 표준화와 유사한 식으로 mini-batch에 적용해 평균 0, 단위 표준 편차 유지를 시도합니다.
식으로 가볍게 살펴봅시다.

![batchnorm_algorithm](https://i.imgur.com/ImAZVCQ.png)

정말 위 식에 나타난 그대로 이 방법은 다음과 같은 방식으로 이루어집니다.

1. mini-batch의 평균
2. mini-batch의 분산과 표준 편차
3. normalize (엡실론을 제외하면 Standardization)
4. 스케일 조정 및 분포 조정

논문에서는 이 방법의 장점을 다음과 같이 이야기 하고 있습니다.

- propagation에서 parameter의 scale 영향을 받지 않으므로, learning rate를 크게 설정할 수 있다.
- weight regularization과 Dropout(이 방법과 성능이 같지만 느림)을 제외할 수 있어, 학습 속도를 향상시킬 수 있다.

> 아래부터는 Batch Normalization을 BN으로 표기합니다.

### 2.2 하지만 또 한편으로는...

이 간단한 방법은 CNN의 성능을 매우 크게 향상시켰습니다.

하지만 [최근 논문](https://arxiv.org/pdf/1805.11604.pdf)에서는 BN의 성공은 Internal Covariance Shift과 상관이 없다고 주장합니다. 그렇다면 왜 성능을 보인건가에 대해 의문을 품을 수 있습니다.

BN의 방법이 효과적인 이유는 직관적인 논리로 다음과 같이 생각할 수 있습니다.

- 신경망에서 가중치의 변경은 그와 연결된 layer에 영향을 미칩니다.
- 이는 가중치의 변화는 매우 복잡하게 신경망에 영향을 미칩니다.
- 그렇기 때문에 Vanishing 또는 Exploding을 막기 위해 small learning rate를 사용하거나, 활성화 함수 사용을 선택합니다.
- Batch Normalization는 최적화를 하기 쉽게 만들기 위해 네트워크를 다시 매개변수화 시킵니다.
- 이 방법은 모든 레이어의 평균, 크기, 활성화함수 를 독립적으로 조정할 수 있게 만듭니다.
- 조정들은 가중치가 미치는 영향을 규제합니다.

> 더 자세한 내용은 다음의 [글](http://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/)을 참고하면 좋을 것 같습니다.

딥러닝의 많은 방법들이 그렇듯, 복잡한 네트워크 속에서 미치는 영향들을 증명을 하기는 어렵습니다.
하지만 결론적으로 어떤 방향이든 BN의 효과는 분명합니다.

- loss surface를 보다 쉽게 찾을 수 있고
- 최적화를 쉽게 만들며
- higher learning rate를 사용할 수 있게 만들고
- 여러 작업에서 모델 성능을 향상시킵니다.

성능면에서 훌륭하지만, 이 방법이 완벽하다고 할 수는 없습니다. 왜 그런지 좀 더 알아보겠습니다.

## 3. Batch Normalization의 문제점

다양한 normalization에서 데이터를 표준화시키는 가장 좋은 방법은 전체 데이터에 대해 평균과 분산을 사용하는 것입니다.
하지만 각 계층에 대해 계속 연산을 한다면 너무 많은 자원을 사용합니다.
그렇기에 다른 효율적인 방법으로 우리가 구하고 싶은 (각 계층 전체)평균과 분산에 근사한 값을 찾아야 합니다.

BN에서는 mini-batch에서 평균과 분산을 사용합니다.
방법론적으로 보았을 때, 약간의 오차는 존재할 수 있으나 괜찮은 추정 방법입니다.

하지만 이 방법이 문제가 생기는 몇 상황이 있을 수 있습니다. 다음은 상황 예시 2가지입니다.

### 3.1 Small batch size

극단적인 상황을 가정해봅시다. batch size가 1일 때는 어떤 상황이 발생할까요.
이 경우에는 분산이 0이므로 정규화를 할 수 없습니다.

이 외의 상황에서도 batch가 작다면 정규화의 값이 기존의 값과 매우 다른 양상을 가지며 훈련에 부정적인 영향을 미칠 가능성이 큽니다.
그렇기에 이 방법에서는 batch size에 하한이 존재합니다.

### 3.2 RNN의 반복 연결

RNN에서는 각 단계마다 서로 다른 통계치를 가집니다. 이는 즉 매 단계마다 레이어에 별도의 BN을 적용해야 합니다. 이는 모델을 더 복잡하게 만들며, 계속 새롭게 형성된 통계치를 저장해야한다는 점입니다. 이 상황에서 BN은 매우 비효율적인 방법처럼 보입니다.

이런 문제 때문에 평균과 분산을 예측하는 BN 외의 정규화 방법을 더 이야기해보도록 하겠습니다.

## 4. 대안책 : 많은 Normalization 방법들

![group-norm](https://i.imgur.com/BTQ0zj5.png)

### 4.1 Weight Normalization

[논문 자료](https://arxiv.org/pdf/1602.07868.pdf)

**Weight Normalization** 은 mini-batch를 정규화하는 것이 아니라 layer의 가중치를 정규화합니다.
Weight Normalization은 레이어의 가중치 $w$를 다음과 같이 재매개변수 시킵니다.

> Weight Normalization은 WN으로 표기합니다.

$$w = \frac{g}{||v||} v$$

BN과 마찬가지로, 이 방법은 표현(expressiveness)을 줄이지 않고 가중치 벡터의 크기와 방향과 분리합니다.
이는 BN에서 입력값을 표준 편차로 나누어주는 것과 비슷한 효과입니다.

그리고 경사하강법으로 $g, v$를 최적화합니다. 그리고 이는 학습에서 최적화를 쉽게 만듭니다.

WN은 경우에 따라 BN보다 빠릅니다.

CNN의 경우, 가중치의 수는 입력의 수보다 훨씬 작습니다. 이는 BN보다 WN이 연산량이 훨씬 적다는 의미입니다.
BN의 경우에는 입력값의 모든 원소를 연산해야하고, 이미지 등의 고차원 데이터에서 연산량이 매우 많아집니다.
그렇기에 WN이 더 빠른 경우가 생기게 됩니다.

WN은 그 자체만으로 모델 훈련에 도움을 줄 수 있지만, 논문의 저자는 ***mean-only batch normalization*** 과 함께 사용하기를 제안하고 있습니다. 이 방법은 입력을 표준 편차로 나누거나 scale 재조정을 하지 않는 BN입니다.

이 방법은 WN보다 속도가 낮지만, 표준 편차를 계산하지 않으므로 BN보다는 연산량이 적습니다. 저자들은 이 방법이 다음과 같은 이익을 제공한다고 주장합니다.

**1. 활성 값의 평균을 $v$와 독립**

WN은 활성화값의 평균과 레이어의 가중치를 독립적으로 분리할 수 없습니다. 그렇기에 각 레이어의 평균간에 높은 종속성이 발생합니다. 하지만 mean-only batch normalization을 사용하면 이런 문제를 해결할 수 있다고 합니다.

**2. 활성화에 "gentler noise" 추가**

BN의 부작용 중 하나는 mini-batch에서 계산된 노이즈가 많은 추정값을 사용하여, 활성화 값에 확률적인 잡음을 추가한다는 것입니다.
일부 문제에서는 이런 노이즈는 규제에 도움을 줄 수 있습니다. 하지만 강화 학습등의 노이즈에 예민한 도메인에서는 성능을 낮출 뿐입니다.

하지만  mean-only batch normalization과 WN을 사용하면 큰 수의 법칙에 따라 노이즈가 훨씬 정규 분포의 형태를 띄며, 이는 노이즈가 완만하다고 표현할 수 있습니다. 또한 BN과 비교해 훈련 과정에서 노이즈가 적습니다.

CIFAR-10 모델에서 이미지 분류 문제를 해결할 때, WN의 효과가 가장 두드러진다고 합니다. 좀 더 자세한 내용은 논문을 참고하시면 됩니다.

### 4.2 Layer Normalization

[논문 자료](https://arxiv.org/pdf/1607.06450.pdf)

**Layer Normalization** 는 WN에 비해 직관적인 파악이 어렵습니다.

> Layer Normalization은 LN으로 표기합니다.

BN과 LN은 거의 유사한 형태를 지닙니다. 위의 큐브 사진 또는 아래의 사진이 이를 시각화한 모양입니다.

<figure>
 <img src = "https://i1.wp.com/mlexplained.com/wp-content/uploads/2018/01/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88-2018-01-11-11.48.12.png?resize=1024%2C598" alt>
 <figcaption> 이미지 출처 : http://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/ </figcaption>
</figure>

LN은 mini-batch의 feature 수가 같아야합니다.

각각 BN과 LN은 **Batch 차원에서 정규화 / Feature 차원에서 정규화** 라는 차이가 있습니다.

BN의 경우에는 batch 전체에서 계산이 이뤄지며, 각 batch에서 동일하게 계산합니다. 하지만 LN의 경우에는 각 특성에 대하여 따로 계산이 이뤄지며, 각 특성에 독립적으로 계산합니다. (hidden unit)

LN은 batch의 사이즈와 상관이 없습니다. 이는 RNN에서 매우 좋은 성능을 보입니다.

### 4.3 Instance Normalization

[논문 자료](https://arxiv.org/pdf/1607.08022.pdf)

**Instance Normalization** 은 LN과 유사하지만, 한 단계 더 나아가 평균과 표준 편차를 구하여 각 example의 각 채널에 정규화를 진행합니다.

> Instance Normalization은 IN으로 표기합니다.

style transfer을 위해 고안된 Instance Normalization은 network가 원본 이미지와 변형된 이미지가 불가지론(구분할 수 없음)을 가지길 바라며 설계되었습니다.
따라서 이미지에 국한된 정규화이며, RNN에서는 사용할 수 없습니다.

style transfer 또는 GAN에서 BN을 대체하여 사용하며, real-time generation에 효과적입니다.

### 4.4 Group Normalization

[논문 자료](https://arxiv.org/pdf/1803.08494.pdf)

**Group Normalization** 은 Instance Normalization과 유사합니다. 다만 여기서는 채널을 그룹으로 묶어 평균과 표준 편차를 구한다는 점입니다.
Group Normalization은 IN과 LN의 중간이라고 생각할 수 있습니다.
모든 채널이 단일 그룹에 있다면 LN, 각 채널을 다른 그룹에 배치하면 IN입니다.

LN과 instance normalization은 모두 공통적으로 RNN과 style transfer에 효과적이지만, BN에 비해 이미지 인식 면에서는 보다 성능이 좋지 못합니다.

Group Normalization에서는 ImageNet에서 batch size가 32일때 BN과 거의 근접한 성능을 냈고, 그 보다 작은 batch size에 대해서는 더 좋은 성능을 냈습니다. object detection 또는 더 높은 해상도의 이미지를 사용하는 작업(메모리의 제약이 있는 작업)에서 매우 효과적이었습니다.

왜 group normalization이 LN 또는 IN보다 효과가 있는지 생각해봅시다.

LN은 모든 채널이 ***동등하게 중요하다*** 라는 가정하에 정규화(평균 계산)가 진행이 됩니다. 하지만 Image의 경우 이미지 가장자리와 중심부의 중요성은 다르다는 것을 경험적으로 알고 있습니다. 즉 여러 채널에서 서로 다르게 계산하면 모델에 따라 유연성을 제공해줄 수 있습니다. 또한 이미지의 경우, 각 채널은 독립된 것이 아니므로(IN) 주변 채널을 활용하여 더 정규화를 넓게 적용할 수 있습니다.

### 4.5 Batch Renormalization

[논문 자료](https://arxiv.org/pdf/1702.03275.pdf)

**Batch Renormalization** 은 BN를 small batch size 사용하기 위한 방법입니다. 기본적인 아이디어는 추론 시에 ***BN의 개별 mini-batch 통계값들을 사용되지 않는다*** 는 점입니다. 대신 이 방법은 mini-batch에서 **이동 평균** 을 사용합니다. 이는 개별 mini-batch 보다 실제 평균과 분산 값을 더 잘 예측하기 위한 방법입니다.

훈련에는 왜 이동 평균을 사용하지 않는가는 ***역전파를 수행해야 한다*** 는 것과 관련이 있습니다.
데이터 정규화에 있어 통계를 사용하려면, 이 값을 역전파 시키는 방법에 대해 적절한 방안이 있어야 합니다.

이전 mini-batch의 활성화 통계값을 사용하여 정규화를 진행하는 경우, 역전파 중에 이전 layer가 통계값에 얼마나 영향을 미쳤는가 계산해야합니다. 이런 상호작용을 무시하면 이전 레이어 손실에 아무런 영향을 미치지 않더라도 활성화값이 계속 증가될 수 있습니다.

훈련 과정에서 이동 평균을 사용하면 모든 이전 mini-batch 값의 통계값을 저장해야하므로 연산량이 매우 많아집니다.

논문의 저자는 이동 평균을 사용하면서 통계에 대한 이전 계층의 영향을 고려하도록 제안합니다.
이동 평균을 이용하여 재매개변수화 하는 것이 이 방법의 핵심입니다.

이동 평균의 평균값을 $\mu$, 편차를 $\sigma$라 하고, mini-batch의 평균과 편차를 $\mu_{B}, \sigma_{B}$ 라 한다면 batch renormalization는 다음과 같이 표현할 수 있습니다.

$$\frac{x_i - \mu}{\sigma} = \frac{x_i - \mu_B}{\sigma_B} \cdot r + d, \ where \ r = \frac{\sigma_B}{\sigma}, d = \frac{\mu_B - \mu}{\sigma} $$

즉, 이동 평균 통계값과 mini-batch에서 얻은 통계값으로 BN에 곱하고 더해서 정규화합니다.

BN의 초기화 설정, 빠른 속도 등의 이점을 가지며 BN과 마찬가지로 Batch Renormalization의 성능 또한 배치 크기가 감소 할 때 여전히 저하됩니다. (BN보다 조금 나음)

### 4.6 Batch-Instance Normalization

[논문 자료](https://arxiv.org/pdf/1805.07925.pdf)

**Batch instance normalization** 은 이미지에서 style과 contrast의 차이를 설명하기 위해 IN을 확장한 정규화입니다.

instance normalization의 문제점은 style 정보를 완전히 지운다는 것입니다.
style transfer에는 유용할 수 있으나, weather classification 과 같이 스타일이 중요한 특징일때는 문제가 될 수 있습니다.

즉, 제거해야하는 style 정보의 정도는 작업에 따라 다릅니다. batch-instance normalization은 각 작업과 특성 맵(채널)에 대해 style 정보의 양을 파악하여 이를 처리하려고 합니다.

BN의 output과 IN의 output을 각각 $\hat{x}^{(B)}, \hat{x}^{(I)}$이라고 하면 batch-instance nomalized output($y$)은 다음과 같은 수식으로 표현됩니다.

$$y = (\rho \cdot \hat{x}^{(B)} + (1-\rho) \cdot \hat{x}^{(I)}) \cdot \gamma + \beta$$

식을 보면, batch-instance normalization은 BN과 IN의 보간입니다. balance 변수 $\rho$는 경사하강법을 통해 학습됩니다.

CIFAR-10/100, ImageNet, domain adaption, style transfer 에서 BN을 능가했습니다.
이미지 분류에서는 $\rho$ 값이 0또는 1에 가까워졌는데, 이는 많은 layer가 instance normalization 또는 BN만 사용을 했음을 알 수 있습니다.

layer들은 IN보다 BN을 사용하는 경향이 있는데, 이는 IN이 불필요한 style transfer을 제거하는 방법으로 더 많이 작용하기 때문입니다.
반면에 style transfer에서는 반대의 경향이  나타났는데, 이는 주어진 style이 sytle transfer에서 덜 중요한 것을 의미합니다.

또한 논문의 저자는 batch-instance normalization에서 $\rho$에 높은 learning rate를 적용하면 성능이 향상하는 것을 발견했습니다.

이 정규화 방법은 ***정규화 방법이 경사하강법으로 다른 정규화 방법을 배울 수 있을까*** 라는 의문을 제기하고, 그 다음 정규화에 대한 아이디어를 제시합니다.

### 4.7 Switchable Normalization

[논문 자료](https://arxiv.org/pdf/1811.07727v1.pdf)

최근 많은 정규화 방법은 계속적으로 발전을 이뤄냈습니다. 그렇지만 batch-instance normalization에서 확인한 것과 같이 정규화 작업을 함께 적용하여, 깊이에 따라 다르게 정규화를 적용한다면 성능이 더 나아질 수 있지 않을까요?

위에 링크는 이 질문에 대한 논문입니다. 위 논문은은 BN, instance normalization, LN 등의 사용을 전환가능한 정규화 방법을 제안하고 있습니다. 또한 이에 대한 가중치는 경사하강법을 통해 학습한다고 합니다.

이미지 분류, object detection 등에 있어 BN보다 우수하다고 합니다.

### 4.8 Spectral Normalization

[논문 자료](https://arxiv.org/pdf/1805.07925.pdf)

BN에 대한 대안은 아니지만, 제가 참고한 포스팅에서 소개하고 있어 가져와봤습니다.
GAN 훈련을 향상시키기 위해 판별자의 Lipschitz 상수를 제한합니다.

실험 결과 최소한의 조정으로 GAN의 학습을 개선함을 보였습니다.

## 5. Conclusion

이 외에도 [Streaming Normalization](https://arxiv.org/abs/1610.06160), [Cosine Normalization](https://arxiv.org/abs/1702.05870) 이 있습니다.

가장 오래된 논문이 2015년이라니, 딥러닝이 얼마나 최신 기술인지 느낄 수 있는 공부였습니다. 보다 시간이 많다면 각 정규화에 대한 논문을 자세히 공부해도 좋을 것 같습니다.

## Reference

### blog posts

- [Regularization: 복잡도를 다스리는 법](https://bahnsville.tistory.com/1140)

- @keitakurita : [An Overview of Normalization Methods in Deep Learning](http://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/)

- @keitakurita : [Weight Normalization and Layer Normalization Explained](http://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2/)

- @shuuki4 : [Batch Normalization 설명 및 구현](https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/)


### papers

본문에서 언급된 논문입니다.

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

- [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1602.07868.pdf)

- [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022.pdf)

- [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf)

- [Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models](https://arxiv.org/pdf/1702.03275.pdf)

- [Batch-Instance Normalization for Adaptively Style-Invariant Neural Networks](https://arxiv.org/pdf/1805.07925.pdf)

- [Do Normalization Layers in a Deep ConvNet Really Need to Be Distinct?](https://arxiv.org/pdf/1811.07727v1.pdf)

- [Batch-Instance Normalization for Adaptively Style-Invariant Neural Networks](https://arxiv.org/pdf/1805.07925.pdf)
