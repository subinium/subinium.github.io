---
title : "[GAN] Improved Techniques for Training GANs"
category :
  - ML
tag :
  - GAN
  - ImprovedGAN
  - review
sidebar_main : true
author_profile : true
use_math : true
header:
  overlay_image : https://images.unsplash.com/photo-1538333702852-c1b7a2a93001?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2252&q=80
  overlay_filter: 0.8
published : true
---

GAN 시리즈 시작합니다!!

**ImprovedGAN**으로 알려져있고, 이미 3000회 이상 인용된 논문으로 GAN을 훈련하는 데 있어 여러가지 인사이트를 줄 수 있는 논문입니다. 여러 참고자료와 함께 같이 해석 및 정리 해보겠습니다.

2016년도 논문이니 GAN 초기 논문이기도 합니다. 

> 경험적인 방법이라, 아이디어가 핵심인 논문입니다.

- [Paper](https://arxiv.org/pdf/1606.03498.pdf)
- [Code with Theano](https://github.com/openai/improved-gan)

## 1. Introduction 

GAN 훈련의 핵심은 다음과 같습니다.

> training GANs requires finding a **Nash equilibrium** of a non-convex game with continuous, highdimensional parameters

**내시 균형(Nash Equibrium)** 이란 게임 이론에서 경쟁자 대응에 따라 최선의 선택을 하면 설로가 자신의 선택을 바꾸지 않는 균형상태를 의미합니다. [wikipedia](https://ko.wikipedia.org/wiki/%EB%82%B4%EC%8B%9C_%EA%B7%A0%ED%98%95)

GAN에서는 생성자(Generator)와 판별자(Discriminator)라는 두 모델이 서로 경쟁하는 상황을 의미하기도 합니다. 

하지만 보통 이 Nash Equibrium 상태를 찾기 위한 과정에서 모델은 실패합니다. 이 논문은 GAN의 수렴을 위해 할 수 있는 몇 가지 테크닉을 소개하고 있습니다.

## 2. Main Idea

핵심적인 아이디어는 총 5가지 입니다.

### 2.1 Feature Matching

**Feature Matching** 은 현재 Disciriminator에 새로운 목표를 지정하여 오버트레이닝을 방지하고, GAN의 불안정성(insatbility)을 해결합니다.

Generator에서 생성한 분포가 실제 데이터의 분포를 matching 시키기 위해  Discriminator 중간층의 activation 함수를 이용합니다.

단순하게 진짜/가짜를 나누는 방식이 아닌, ***진짜와 같은 feature를 가지고 있느냐?*** 라는 방식으로 훈련을 진행하는 것입니다.

이를 위해 새로운 손실함수 다음과 같은 방식으로 정의하고 사용합니다.

$$||E_{x \sim p_data} f(x) - E_{z \sim p_z(z)}f(G(z))||^2_2$$

$f(x)$는 Discriminator의 중간 층 activation 함수입니다. 식을 이해해보면 Discriminator 중간층의 output이 생성에 필요한 하나의 **특징(feature)**이며, 이게 random sampling된 z에 대해 **분포가 비슷한지(matching)** 살펴보는 것입니다.

*G가 목표하는 통계치에 도달하는지는 확신할 수 없지만, 경험적으로 불안정한 GAN에 대해 효과적이다.* 라고 이야기 하고 있습니다.

### 2.2 Minibatch discrimination

GAN이 실패하는 경우 중 하나는 Generator가 동일한(유사한) 출력을 하게 parameter가 세팅되는 경우입니다. Generator는 Discriminator를 속이기만 하면 되기 때문에 이런 일이 발생합니다. 

> Vanila GAN 등에서 실험해보면 초기 noise에 따라 Loss가 급격하게 줄어들고 결과를 확인하면 이상한 noise만 출력하는 것을 확인할 수 있습니다.

논문에서는 이 문제는 Discriminator가 각 example을 개별로 처리하기 때문에 출력간의 관계를 고려하지 않기 때문이라고 이야기하고 있습니다. 그래서 배치(batch) 안에서 다른 데이터간의 관계를 고려하도록 설계하는 방법입니다.

Discriminator는 여전히 실제/생성 데이터를 분류하는 일을 하고, minibatch의 정보를 side information으로 사용할 수 있습니다. (개별 샘플이 minibatch내의 다른 샘플들과의 유사도(L1 norm)를 계산하여 합치고, 이를 판별에서 추가적인 정보로 사용하게 됩니다.)

Minibatch Discriminator를 사용하면 시각적으로 매력있는 샘플을 feature matching보다 잘 생성합니다.

하지만 feature matching은 semi-supervised learning에서는 더 성능이 좋았다고 합니다.

### 2.3 Historical averaging

dicrimnator와 generator의 손실함수 모두에 $| |\theta - \frac{1}{t} \sum^t_{i=1} \theta[i] | |^2$를 추가합니다. 

$\theta$는 모델 파라미터를 나타내고, $\theta[i]$는 $i$번째 학습 과정에서의 파라미터를 의미합니다.

### 2.4 One-sided label smoothing

Label smoothing은 0과 1 타겟 대신 0.9, 0.1 등의 smoothed value로 classifier을 훈련하는 방법입니다. 구체적으로는 positive target을 $\alpha$, negative target을 $\beta$ 로 두는데, 여기서 negative data가 더 좋은 방향으로 생성되는 것을 위해 $\beta$는 0으로 두게 됩니다. (One-sided)

### 2.5 Virtual batch normalization 

Mini batch의 다른 값들의 영향을 많이 받는 것을 방지하기 위해 고정된 배치(reference batch)를 이용하는 방식입니다. 

reference batch는 학습 초기에 한번 선별되어 학습이 진행되는 동안 변하지 않습니다. 그리고 이 값을 이용하여 normalize합니다.

하지만 2개의 minibatch를 forward propagation하는 것은 느리기 때문에 generator에서만 진행합니다.

## 3. 개인적인 후기

- 영어 표현 자체가 매우 어려워 내용 이해가 매우 어려웠습니다. 그런 만큼 제가 잘못해석한 부분이 있을 수 있으니 틀린 번역 또는 해석이 있다면 댓글로 지적해주시면 감사하겠습니다.

- 이미지에서는 성능이 엄청 높아지지는 않았지만, 다른 task에서는 성능이 있지 않을까? 라는 생각을 하며 읽었습니다.

## Reference

- [Anomaly Detection with GAN](http://incredible.ai/deep-learning/2018/02/17/AnoGAN/#feature-matching)

- [From GAN to WGAN](https://github.com/yjucho1/articles/blob/master/fromGANtoWGAN/readme.md)

- [Generative Models Part 2: ImprovedGAN,InfoGAN,EBGAN](https://taeoh-kim.github.io/blog/generative-models-part-2-improvedganinfoganebgan/)