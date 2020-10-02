---
title: "[XAI] Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI 리뷰"
category:
  - ai
tag:
  - XAI
  - Machine Learning
  - Deep Learning
  - Data Fusion
  - Interpretability
  - Comprehensibility
  - Transparency
  - Privacy
  - Accountability
  - Responsible
author_profile: true
header:
  overlay_image: https://images.unsplash.com/photo-1517409091671-180985f2ca15?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjExMDk0fQ&auto=format&fit=crop&w=1350&q=80
  overlay_filter: 0.7
published: False
---

- Interpretability
  - 의사 결정의 공정성을 보장하는 데 도움
  - 예측을 변경할만한 잠재적인 (적대적) 혼란을 강조하여 견고성을 제공
  - 의미있는 변수만 결과 추론을 하도록 보험 역할을 해줄 수 있음 (인과 관계의 의미성 보장)

## 2. Explainability: What, Why, What For and How?

### 2.1. Terminology Clarification

AI와 XAI에서 사용되는 설명과 관련된 전문 용어의 차이점을 살펴봅시다.

생각보다 조금씩 다르며, 기본적으로는 Understandability를 지향한다고 볼 수 있습니다.

- **이해가능성(Understandability)** : 내부의 구조나 처리 알고리즘을 설명할 필요없이 인간이 모델의 기능을 이해할 수 있도록 하는 모델의 특성
- **이해가능성(Comprehensibility)** : 학습 알고리즘이 학습 된 지식을 사람이 이해할 수 있는 방식으로 표현하는 능력 (청중에 따라 다름)
- **해석가능성(Interpretability)** : (모델의 결정을) 인간에게 이해할 수 있는 용어로 의미를 설명 또는 제공하는 능력
- **설명가능성(Explainability)** : 인간과 의사 결정자 사이의 인터페이스로서 설명이라는 개념과 관련이 있으며, 동시에 의사 결정자의 정확한 대리이자 인간이 이해할 수 있습니다.
- **투명성(Transparency)** : 모델을 그 자체로 이해할 수 있다면 _투명하다_ 라고 여길 수 있습니다.
  - 시뮬레이션 가능한 모델(simulatable model)
  - 분해 가능한 모델(decomposable model)
  - 알고리즘적으로 투명한 모델(algorithmically transparent model)

### 2.2. What?

우선 **Explainable Artificial Intelligence(이하 XAI)** 에 대한 정의는 매우 모호했습니다. Interpretability과 Explainability의 차이 또한 문서에서 이야기한 바가 없습니다.

그래서 이에 관한 첫 레퍼런스로 XAI는 D.Gunning에 의해 다음과 같이 정의가 되었습니다.

> "XAI will create a suite of machine learning techniques that enables human users to understand, appropriately trust, and effectively manage the emerging generation of artificially intelligent partners"

이 정의는 이해(understanding)와 신뢰(trust)를 결합합니다. 하지만 인과관계, 전달 가능성, 정보성, 공정성 및 신뢰도와 같은 Interpretable AI에서 말하는 필요성에 대한 정보들을 포함시키지 않고 있습니다.

그렇기에 위의 정의는 용어에 대한 정의로는 부족하다고 저자들은 이야기합니다.
XAI에 앞서 설명에 대한 정의를 Cambridge Dictionary of English Language에서 살펴보면 다음과 같습니다.

> explanation : the details or reasons that someone gives to make something clear or easy to understand

이제 여기서부터 모호해집니다. 그 모호함은 다음과 같은 부분에서 살펴볼 수 있습니다.

1. 설명에 사용되는 세부적인 내용과 이유는 청중에게 전적으로 의존한다.
2. 설명이 그 개념을 잘 전달했는지도 전적으로 청중에게 의존한다.

즉, 모델의 Expalinabililty에 청중에 대한 의존성을 포함시켜야 합니다.

위의 정의를 model을 추가하여 다음과 같이 정의해볼 수 있습니다.

> Given a certain audience, explainability refers to the details and reasons a model gives to make its functioning clear or easy to understand.

하지만 이해의 정도를 명확하게 측정하는 일이 어렵기에 청중의 이해를 완벽하게 수치적으로 접근하는 것은 어렵습니다.

그러나 모델 내부를 어느 정도까지 설명할 수 있는지 측정하는 것은 객관적으로 다루어질 수 있습니다.
모델의 복잡성을 줄이거나 출력을 단순화하는 등의 모든 방식을 XAI의 접근 방식으로 고려해야 합니다.

이런 결과들을 종합했을 때, XAI는 다음과 같이 정의됩니다.

> Given an audience, an explainable Artificial Intelligence is one that produces details or reasons to make its functioning clear or easy to understand.

### 2.3. Why?

XAI의 목적은 크게 두 가지입니다.

1. 연구 커뮤니티와 비즈니스 부문 간의 격차
   - 은행, 재무, 보안, 보건과 같은 프로세스에 ML 모델이 완전히 보급되는 것을 막고 있음
   - 자산의 위험이 발생할 수 있는 여러 task에서 규제
2. 지식의 축
   - 결과와 성과만을 목적으로 하는 AI 및 ML 경향
   - 모델 개선과 실용성을 위한 발판

![XAI reason](https://i.imgur.com/HO3Zlfv.png)

### 2.4. What for?

![Goal](https://i.imgur.com/Wh5LflB.png)

- **Trustwothiness**
  - 모델이 주어진 문제를 직면했을 때, 의도한 대로 행동하는가?
  - 설명가능한 모델은 신뢰성이 있어야 하나, 신뢰할 수 있는 모델이 설명가능한 것은 아님
- **Causality**
  - 데이터 변수들 사이의 인과관계
  - 인과성을 증명하기 위해서는 광범위한 사전 지식이 필요
  - 학습한 데이터 간의 상관관계만 발견하므로 인과관계를 밝히는데 불충분
  - 하지만 상관관계는 인과관계의 부분, 사용 가능한 데이터 내에서 최초의 직관 제공
- **Transferability**
  -
- **Informativeness** : 정보
- **Confidence** : 안정성(robustness and stability)에 대한 신뢰도
- **Fariness** : 공정성
- **Accessibility** : 접근성
- **Interactivity** : 상호작용
- **Privacy awareness** : 프라이버시

### 2.5. How?

### 2.5.1. Levels of Transparency in Machine Learning Models

- Simulatability
- Decomposability
- Algorithmic Transparency

### 2.5.2. Post-hoc Explainability Techniques for Machine Learning Models

- Text explanation
- Visual explanation
- Local explanation
- Explanations by example
- Explanations by simplification
- feature relevance explanation

## 3. Transparent Machine Learning Model
