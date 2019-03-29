---
title : Introduction to Activation Function

category :
  - ML
tag :
  - UI/UX
  - custumize
  - jupyter
  - notebook
  - python
  - ppt
sidebar_main : true
author_profile : true
use_math : true
header:
  teaser : https://www.de.digital/DIGITAL/Redaktion/EN/Bilder/Meldungen/2017-04-21-zypries-seven-new-digital-hubs-834x468.jpg?__blob=normal&v=3
  overlay_image : https://www.de.digital/DIGITAL/Redaktion/EN/Bilder/Meldungen/2017-04-21-zypries-seven-new-digital-hubs-834x468.jpg?__blob=normal&v=3
published : true
---

activation을 알아봅시다.

딥러닝을 시작하는 책 또는 문서 대다수는 일부 매개변수에 대한 설명을 생략하거나 가볍게 넘어갑니다. 예를 들면 다음과 같습니다.

- 활성화 함수 : 왜 relu?
- 옵티마이저 : 왜 RMSprop?
- 손실함수 : 왜 categorical_crossentropy?

시간이 날 때마다 이런 가볍게 넘어가는 부분들을 정리해보고자 합니다.
딥러닝이 계속 발전될 수 있는 이유는 이런 부분적인 함수나 방법들의 개선에서 부터 시작했다고 생각합니다.

Ian Goodfellow 가 GAN을 생각한 것처럼 공부를 하다보면 좋은 인사이트를 얻을 수 있지 않을까요?

저번에는 normalization이었다면 이번에는 activation입니다. 시작해보겠습니다!

## 1. Activation function의 역할

**활성화 함수** 라고 번역되는 Activation function은 **신경망의 출력을 결정하는 식** 입니다.

![활성화 함수 이미지](https://upload.wikimedia.org/wikipedia/commons/6/60/ArtificialNeuronModel_english.png)

신경망에서는 뉴런(노드)에 연산 값을 계속 전달해주는 방식으로 가중치를 훈련하고, 예측을 진행합니다.

각각의 함수는 네트워크의 각 뉴런에 연결되어 있으며, 각 뉴런의 입력이 모델의 예측과 관련되어 있는 지 여부에 따라 *활성화* 됩니다. 이런 활성화를 통해 신경망은 입력값에서 필요한 정보를 학습합니다.

활성화 함수는 훈련 과정에서 계산량이 많고, 역전파(backpropagation)에서도 사용해야 하므로 연산에 대한 효율성은 중요합니다. 그렇다면 이런 활성화 함수의 종류를 살펴보겠습니다.

## 2. Activation 3가지 분류

### 2.1 Binary step function

<figure>
    <img src = "https://missinglink.ai/wp-content/uploads/2018/11/binarystepfunction.png">
    <figcaption>이미지 출처 : https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right </figcaption>
</figure>

**Binary step function** 은 **임계치** 를 기준으로 출력을 해주는 함수입니다. 퍼셉트론(perceptron) 알고리즘에서 활성화 함수로 사용합니다.

$$\sigma(x)=
\begin{cases}
0, & x \le 0 \\
1, & x > 0
\end{cases}$$

이 함수의 경우에 다중 분류 문제와 같은 문제에서 다중 출력을 할 수 없다는 단점이 있습니다.

### 2.2 Linear activation function

<figure>
    <img src = "https://missinglink.ai/wp-content/uploads/2018/11/graphsright.png">
    <figcaption>이미지 출처 : https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right </figcaption>
</figure>


**Linear activation function** 는 말그대로 선형 활성화 함수입니다.

$$h(x) = cx, \ c\ is\  constant $$

입력 값에 특정 상수 값을 곱한 값을 출력으로 가집니다. 다중 출력이 가능하다는 장점이 있지만, 다음과 같은 문제점을 가집니다.

**1. backpropagation 사용이 불가능합니다.**

기본적으로 **역전파** 는 활성화함수를 미분하여 이를 이용해 손실값을 줄이기 위한 과정입니다. 하지만 선형함수의 미분값은 상수이기에 입력값과 상관없는 결과를 얻습니다.

그렇기 때문에 예측과 가중치에 대한 상호관계에 대한 정보를 얻을 수 없습니다.

**2. 은닉층을 무시하고, 얻을 수 있는 정보를 제한합니다.**

흔히 딥러닝을 ***구겨진 공을 피는 과정*** 이라고 표현을 합니다. 이는 복잡한 입력을 신경망, 활성화 함수를 이용해 정보를 컴퓨터가 이해하기 쉽게 변환하는 딥러닝의 과정을 비유한 의미입니다.

<figure>
    <img src = "https://i.imgur.com/7SzLe5T.jpg">
    <figcaption> 입력을 추상화하면 다음과 같이 생각할 수 있습니다. 딥러닝은 상상력이 필요한 작업이죠.</figcaption>
</figure>

활성화 함수를 여러 층을 통해 얻고자 하는 것은 필요한 정보를 얻기 위함 입니다. 하지만 선형함수를 여러번 사용하는 것은 마지막에 선형함수를 한번 쓰는 것과 같습니다.

$h(x) = cx$ 일때, $h(h(h(x))) = c'x$이기 때문입니다.

### 2.3 Non-linear activation function

이제 위의 두 종류의 활성화 함수의 단점때문에 활성화 함수는 비선형 함수를 주로 사용합니다.

최근 신경망 모델에서는 거의 대부분 비선형 함수를 사용합니다. 입력과 출력간의 복잡한 관계를 만들어 입력에서 필요한 정보를 얻습니다. 비정형적인 데이터에 특히 유용합니다. (이미지, 영상, 음성 등의 고차원 데이터)

비선형 함수가 좋은 이유는 선형 함수와 비교해 다음과 같습니다.

1. 입력과 관련있는 미분값을 얻으며 역전파를 가능하게 합니다.
2. 심층 신경망을 통해 더 많은 핵심 정보를 얻을 수 있습니다.

## 3. Non-linear Activation 종류

식과 함수 형태 등의 정리는 [위키피디아 자료](https://en.wikipedia.org/wiki/Activation_function)를 추천합니다.

다음 함수들은 실제 딥러닝에서 사용되는 함수입니다. 더 많은 종류의 함수가 있지만, [케라스](https://keras.io/layers/advanced-activations/)에서 제공하는 activation을 위주로 목록을 작성했습니다.

추가를 바라는 분이 있다면 댓글로 달아주시면 감사하겠습니다.

### 3.1 Sigmoid

로지스틱(logistic)으로도 불리는 sigmoid 함수는 s자 형태를 띄는 함수입니다.

![output_5_0](https://i.imgur.com/W9Eg05g.png)

그래프는 살짝 아쉽지만 볼 수 있듯이 입력값이 커질수록 1로 수렴하고, 입력값이 작을수록 0에 수렴합니다.

함수의 식과 미분 값은 다음과 같습니다.

$$\sigma(x) = \frac{1}{1+exp(-x)}$$

$$\sigma'(x)= \sigma(x)(1-\sigma(x))$$

증명은 다음 [stackoverflow](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x/1225116#1225116) 를 참고하면 됩니다.

#### Pros

- 유연한 미분값을 가집니다.
- 출력값의 범위가 (0, 1)로 제한됩니다. 정규화 관점에서 **exploding gradient** 문제를 방지합니다.
- 미분 식이 단순한 형태를 가집니다.

#### Cons

- **Vanishing Gradient 문제가 발생합니다.**

$\frac{\sigma(x) + (1-\sigma(x))}{2} = \frac{1}{2} \ge \sqrt{\sigma(x)(1-\sigma(x))}$

의 식에 따라 미분 값의 범위는 (0, 1/4) 임을 알 수 있습니다. 입력이 아무리 커도 미분 값의 범위는 제한됩니다. 층이 쌓일수록 gradient 값이 0에 수렴할 것이고, 학습의 효율이 매우 떨어지는 것을 직관적으로 알 수 있습니다. 또한 극값으로 갈수록 값이 포화됩니다.

- **출력의 중심이 0이 아닙니다.**

이것이 단점인 이유를 직관적으로 알기는 어렵습니다.
[stackoverflow](https://datascience.stackexchange.com/questions/14349/difference-of-activation-functions-in-neural-networks-in-general)에서 이런 답변을 찾을 수 있었습니다.

간단하게 설명하면 x가 모두 양수로 들어올 경우, gradient의 값이 모두 양수 또는 모두 음수의 형태를 지녀 zigzag 꼴로 학습하며, 이 방법이 비용/효율면에서 좋지 못하다는 것입니다.

>가정 자체가 마음속으로 와닿지는 않지만 그렇다고 합니다.

- exp연산은 비용이 큽니다.

퍼셉트론 등 초기 신경망에 많이 사용했지만 여러 단점 때문에 현재는 많이 사용하지 않는 함수입니다.

### 3.2 Tanh

tanh 또는 hyperbolic tangent 함수는 쌍곡선 함수입니다. 시그모이드 변형을 이용해 사용가능합니다.


![output_5_1](https://i.imgur.com/WviQUML.png)

$$tanh(x) = 2\sigma(2x)-1$$

$$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

$$tanh'(x) = 1 - tanh^2(x)$$

#### Pros

- Zero Centered 입니다.

이 부분에 대한 sigmoid의 단점을 해결할 수 있습니다.

- 다른 장점은 sigmoid와 같습니다.

#### Cons

- center문제를 제외하고 sigmoid와 같습니다.

tanh도 시그모이드와 함께 잘 사용하지 않는 함수입니다.

### 3.3 ReLU

Rectified Linear Unit 함수의 준말로 개선 선형 함수라고 생각할 수 있습니다. 그래프만 봐도 명칭을 이해할 수 있습니다. CNN에서 좋은 성능을 보였고, 현재 딥러닝에서 가장 많이 사용하는 활성화 함수 중 하나입니다.

실제 뇌와 같이 모든 정보에 반응하는 것이 아닌 일부 정보에 대해 무시와 수용을 통해 보다 효율적인 결과를 낸다고 생각할 수 있습니다.

![output_5_2](https://i.imgur.com/sWhVeLg.png)

$$f(x) = max(0,x)$$

$$f'(x)=
\begin{cases}
0, & x \le 0 \\
1, & x > 0
\end{cases}$$

#### Pros

- **연산이 매우 빠릅니다.**

함수의 원형을 통해 알 수 있듯, 연산은 비교연산 1회를 통해 함숫값을 구할 수 있습니다. 수렴속도 자체는 위의 두 함수보다 6배 이상 빠릅니다.

- **비선형 입니다.**

모양 자체는 선형같지만, 이 함수는 비선형 함수입니다. 도함수를 가지며, backpropagtion을 허용합니다.
또한 위에서 언급한 바와 같이 정보를 효율적으로 받습니다.

#### Cons

- **Dying ReLU**

입력값이 0또는 음수일때, gradient값은 0이 됩니다. 이 경우 학습을 하지 못합니다.
데이터의 희소성은 ReLU를 효과적으로 만들어줬고, 이것이 ReLU의 단점이기도 합니다.

이 문제를 해결하기 위해 다양한 유사함수가 만들어집니다. 유사함수는 아래에 소개되어 있습니다.

### 3.4 softmax

MNIST 등의 기본적인 다중 분류 문제를 해결하신 분들에게는 익숙한 함수입니다.

softmax함수는 입력받은 값을 0에서 1사이의 값으로 모두 정규화하며, 출력 값이 여러개입니다. 출력 값의 총합은 항상 1이 되는 특징을 가집니다.

![softmax](https://wikimedia.org/api/rest_v1/media/math/render/svg/6d7500d980c313da83e4117da701bf7c8f1982f5)

출력 층에 많이 사용합니다. 식은 다음과 같습니다.

#### Pros

- **다중 클래스 문제에 적용 가능합니다.**

- 정규화 기능을 가집니다.

#### Cons

- 지수함수를 사용하여 오버플로 발생이 가능합니다. (분모분자에 C를 곱해 이를 방지)

---

여기까지는 제가 지금까지 흔하게 사용했던 활성화 함수입니다. 이제 조금 더 다양한 활성화 함수를 보겠습니다.

### 3.5 Leaky ReLU

Leaky의 의미는 *새는, 구멍이 난* 입니다. ReLU에서 Dying ReLU 문제를 해결하기 위해 만든 함수입니다.
음수부에 매우 작은 상수를 곱한 ReLU입니다. 범위가 작아 그래프는 거의 유사하게 그려졌습니다.

![output_5_3](https://i.imgur.com/0crn3nu.png)

$$f(x) = max(0.01x, x)$$

$$f'(x)=
\begin{cases}
0.01, & x \le 0 \\
1, & x > 0
\end{cases}$$

#### Pros

- **Dying ReLU문제를 방지합니다.**
- 연산이 (여전히) 빠릅니다.
- ReLU보다 균형적인 값을 반환하고, 이로 인해 학습이 조금 더 빨라집니다.

#### Cons

- ReLU보다 항상 나은 성능을 내는 것은 아니며, 하나의 대안책으로 추천합니다.

### 3.6 Parametric ReLU (PReLU)

Leaky ReLU와 거의 유사하지만 상수를 원하는 값으로 설정합니다. ReLU와 거의 유사합니다.

$$f(\alpha, x) = max(\alpha x, x)$$

$$f'(\alpha, x)=
\begin{cases}
\alpha, & x \le 0 \\
1, & x > 0
\end{cases}$$

#### Pros

- 문제에 따라 유동적으로 설정할 수 있다는 장점이 있습니다.
- Leaky ReLU와 같습니다.

#### Cons

- 문제에 따라 다른 상수값을 설정해야 한다는 단점이 있습니다.
- Leaky ReLU와 같습니다.

### 3.7 ELU

**ELU** 는 Exponential Linear Unit을 의미합니다. 음수일 때 exp를 활용하여 표현합니다.

![output_5_4](https://i.imgur.com/jd5Jrf6.png)

그래프에서 $\alpha$를 2로 설정한 결과입니다.

$$f(\alpha, x)=
\begin{cases}
\alpha(e^x-1), & x \le 0 \\
x, & x > 0
\end{cases}$$

$$f'(\alpha, x)=
\begin{cases}
f(\alpha, x) + \alpha, & x \le 0 \\
1, & x > 0
\end{cases}$$

여기에 scale 상수 $\lambda$를 곱해주면 Scale Exponential Linear Unit(SELU) 함수입니다.

#### Pros

- ReLU의 모든 장점을 포함합니다.
- Dying ReLU 문제를 해결했습니다.

#### Cons

- exp 함수를 사용하여 연산 비용이 추가적으로 발생합니다.
- 큰 음수값에 대해 쉽게 포화됩니다.

### 3.8 Swish

다음 [논문](https://arxiv.org/pdf/1710.05941v1.pdf)(2017)에서 소개된 활성화 함수입니다. SiLU(Sigmoid Linear Unit)라고도 불립니다.

![output_5_5](https://i.imgur.com/tIirjwM.png)

시그모이드 함수에 입력값을 곱한 함수로 특이한 형태를 가집니다.

$$f(x) = \frac{x}{1+e^{-x}} = x \cdot \sigma(x)$$

$$f'(x) = f(x) + \sigma(x)(1-f(x))$$

논문을 잠시 살펴본 결과 다음과 같은 이미지를 찾을 수 있었습니다.

![논문 발췌](https://i.imgur.com/23zUWx0.png)

논문에 따르면 2차원에서 확인했을 때, linear 또는 ReLU보다 훨씬 부드러운 형태를 가집니다.

ReLU 및 다른 활성화 함수를 대체하기 위해 만든 함수입니다. 논문에서는 CIFAR 등의 예시에서 실험한 결과, ReLU 및 다른 활성화 함수보다 좋은 성능을 가진다고 합니다.

### 3.9 softplus

softplus는 sigmoid 함수의 적분값 입니다. 다른 말로 하면, 이 함수의 도함수의 값은 sigmoid 함수입니다.

![output_5_6](https://i.imgur.com/U25SJPS.png)

$$f(x) = ln(1+e^x)$$

$$f'(x) = \frac{1}{1+e^{-x}} = \sigma(x)$$

#### Pros

- ReLU의 경우 경계에서 매우 뾰족한 형태를 띄지만 Softplus는 결정 경계가 매우 부드럽습니다. (smooth decision boundary)

#### Cons

- 부드러운 경계로 인해 정규화 능력이 비교적 부족합니다.

- ReLU 보다 속도면에서도 느리기 때문에 잘 사용하지 않습니다.

### 3.10 softsign

tanh함수를 대체하기 위해 만든 활성화 함수입니다.
하지만 tanh보다 적게 사용됩니다. 전반적으로 안쓰는 함수인 것 같습니다.

![output_5_7](https://i.imgur.com/u7YxFZB.png)

$$f(x) = \frac{x}{1+|x|}$$

$$f'(x) = \frac{1}{(1+|x|)^2}$$

### 3.11 Thresholded ReLU

이 함수는 임계치를 설정한 ReLU입니다. 위키피디아에는 나오지 않았지만 Keras에서 사용할 수 있는 함수 중 하나입니다. 다음과 같은 형태를 가집니다.

$$f(x, \theta)=
\begin{cases}
0, & x \le \theta \\
x, & x > \theta
\end{cases}$$

$\theta$가 0일때 ReLU와 같습니다. 일부 문제에 있어 ReLU 대안책의 한 방법으로 사용합니다.

### 3.12 Maxout

softmax와 마찬가지로 출력이 여러개로 이루어진 활성화 함수입니다. 효과가 매우 좋은 활성화 함수라고 합니다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/eeda24441c3129f46adeeac876c6fe3dfffb73c9)

이 함수에 대한 소개 글 중에 매우 좋은 자료가 있어 링크를 올립니다.

- [[라온피플 : Machine Learning Academy_Part VI. CNN 핵심 요소 기술] 4.Maxout](http://blog.naver.com/laonple/220836305907)

#### Pros

- ReLU의 장점을 가집니다.
- 성능이 매우 좋습니다.
- Dropout과 함께 사용하기 좋은 활성화 함수입니다.

#### Cons

- 계산량이 많고 복잡합니다.

### 3.13 그 외

소개하지 않은 함수는 아직 많습니다. 예를 들면 다음과 같은 함수가 있습니다. 그래프가 이뻐서 가져와봤습니다.

![SoftExponential](https://i.imgur.com/973Ezth.png)

## 4. Conclusion

> 아래에도 Reference에도 언급했지만 좀 더 코드로 구현해보고 싶으신 분은 다음 포스팅을 추천힙니다. [Coding Neural Network — Forward Propagation and Backpropagtion](https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76)

어떤 함수가 제일 좋다고는 할 수 없지만, 자주 사용하는 활성화 함수는 이미 일부 정해져있습니다. 하지만 ***모든 문제에 최적화된 함수는 없다*** 는 것이 포인트입니다.

어떤 문제에 있어서는 새로운 활성화 함수가 유용한 케이스가 존재할 것이고, 간단한 아이디어만으로 성능을 향상 시킬 수 있다고 생각합니다. 그런만큼 딥러닝에서는 직관을 키우는 것이 매우 중요하다고 생각합니다.

활성화 함수를 포함하여 딥러닝을 공부할 때는 생각을 항상 열어두고 다양한 가능성을 제시하는 연습을 해야겠습니다.

## 5. 부록

본문의 그래프는 다음과 같은 코드로 작성되었습니다.

``` python
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

scope = np.linspace(-5,5)

# draw function axes[0] is f, axes[1] is df
def draw(f, df, scope):
    fig = plt.figure
    fig, axes = plt.subplots(1,2, figsize=(12, 5))
    axes[0].plot(scope, f(scope))
    axes[1].plot(scope, df(scope))
    plt.axis('equal')
    plt.show()

# activation function to draw

# Sigmoid
def sig(x):
    return 1/(1+np.exp(-x))

def dsig(x):
    return sig(x) * (1-sig(x))

# tanh
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1-tanh(x)**2

# ReLU : Thresholded ReLU
def ReLU(x):
    return np.maximum(0,x)

def dReLU(x):
    return np.maximum(0,np.abs(x)/(x))

# LeakyReLU : PReLU
def LeakyReLU(x):
    return np.maximum(0.01*x,x)

def dLeakyReLU(x):
    return np.maximum(0.01,np.abs(x)/(x))

# ELU
def ELU(x):
    return np.maximum( 2*(np.exp(x) - 1) * abs(x)/-x , x )

def dELU(x):
    return np.maximum((ELU(x) + 1)  * abs(x)/-x , 1 )

# swish
def swish(x):
    return x*sig(x)

def dswish(x):
    return swish(x) + sig(x)*(1-swish(x))

# softplus

def softplus(x):
    return np.log(1+np.exp(x))

def dsoftplus(x):
    return sig(x)

# softsign

def softsign(x):
    return x/(1+np.abs(x))

def dsoftsign(x):
    return 1/((1+np.abs(x))**2)


active_func = [sig, tanh, ReLU, LeakyReLU, ELU, swish, softplus, softsign]
active_dfunc = [dsig, dtanh, dReLU, dLeakyReLU, dELU, dswish, dsoftplus, dsoftsign]

for f, df in zip(active_func, active_dfunc):
    draw(f,df,scope)

```


## Reference

- [Keras Official Document : activation / advanced activations ](https://keras.io/layers/advanced-activations/)

- [Activation Function - wikipedia](https://en.wikipedia.org/wiki/Activation_function)

- [7 Types of Neural Network Activation Functions: How to Choose?](https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/#!#typesofactivationfunctions)

- [Fundamentals of Deep Learning – Activation Functions and When to Use Them?](https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/)

- [A Practical Guide to ReLU](https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7)

- [Coding Neural Network — Forward Propagation and Backpropagtion](https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76)

- [[라온피플 : Machine Learning Academy_Part VI. CNN 핵심 요소 기술] 4.Maxout](http://blog.naver.com/laonple/220836305907)

- @reniew : [딥러닝에서 사용하는 활성화 함수](https://reniew.github.io/12/)

- @jleewebblog : [딥러닝에서 활성 함수 가이드 (Activation function guide in deep learning)](https://jleewebblog.wordpress.com/2016/10/26/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%97%90%EC%84%9C-%ED%99%9C%EC%84%B1-%ED%95%A8%EC%88%98-%EA%B0%80%EC%9D%B4%EB%93%9C-activation-function-guide-in-deep-learning/)

- @김콜리(kolikim)님 : [#2-(2) 신경망 : 활성화 함수](https://kolikim.tistory.com/15)
