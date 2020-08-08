---
title: "[HCI] Fluid Interaction for Information Visualization 리뷰"
category:
  - hci
tag:
  - interactive
  - data visualization
  - paper review
author_profile: true
header:
  overlay_image: https://images.unsplash.com/photo-1518837695005-2083093ee35b?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2550&q=80
  overlay_filter: 0.7
---

시각화와 인터랙티브의 균형을 찾아

> 잘못된 해석이 있을 수 있으며, 2011년 논문임을 감안하고 읽으면 좋을 것 같습니다. 제 입맛에 맞게 내용을 추가/삭제 했습니다.

저는 보통 Python으로 시각화를 진행하면 2가지 도구를 메인으로 작업합니다.

1. matplotlib
2. plotly

단순하게 나누자면 정적 시각화와 인터랙티브(상호작용 가능한) 시각화입니다.
matplotlib도 인터랙티브를 제공하지만, 생산성 측면에서는 plotly에 한참 못미치기 때문입니다.

하지만 생각보다 인터랙티브 시각화가 꼭 좋은 것만은 아닙니다.
오히려 더 시각적으로 혼란을 주는 경우도 있고, 정적만으로도 충분한 경우가 있습니다.

그래서 이런 인터랙티브한 시각화를 좀 더 체계적으로 정리한 논문(Niklas et, 2011)이 있어 소개합니다.

- [Fluid Interaction for Information Visualization (2011)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.309.2857&rep=rep1&type=pdf#:~:text=the%20physical%20property%20of%20a,several%20of%20the%20following%20properties%3A&text=Promotes%20flow%3A%20The%20interaction%20should,promote%20staying%20in%20the%20flow.)

## Abstract

시각화에서 **상호작용(interaction)** 은 사용자와 데이터의 대화를 촉진하고, 실제로 이해와 통찰력을 제공합니다.

하지만 시각적인 측면과 상호적인(대화적인) 측면은 균형을 잡기 쉽지 않습니다.

한 가지 이유는 상호작용이 설계, 수량화 및 평가가 어려운 무형의 개념이기 때문입니다

그리고 비주얼 디자인(visual design)과 다르게 개발자와 연구자가 좋은 방법으로 인터랙티브하게 시각화하는 예시가 있습니다.

이 논문에서는 이런 최상의(best-in-class) 예시들에서 미래의 디자이너 및 연구원을 위한 실용적인 디자인 가이드라인을 만들어 문제를 해결하고자 합니다.

그리고 저자들은 이 개념을 **fluid interaction** 라고 하며, direct manipulation, embodied interaction paradigms, 'flow'의 심리적 개념, Norman 실행-평가 모델을 활용하여 정의를 시도합니다.

## Introduction

> 고전적인 HCI 1.0에서 HCI 2.0으로 넘어가는 단계의 논문입니다. [(HCI 개론(김진우 저) 1장 참고)](/_posts/hci/book/2020-08-02-introduction-to-hci-1.md)

본 논문에서는 편리한 사용뿐만 아니라, 매끄럽고 미적으로 아름다운 InfoViz 툴을 위한 가이드라인을 제시하고자 합니다.
fluid interaction의 정의를 통해 연구자와 실무자 모두에게 통용할 수 있는 가이드라인을 제안하고, 이러한 상황을 해결하고자 합니다.
그리고 fluidity가 의사 결정 프로세스를 효율적이고, 즐거운 경험으로 변환할 수 있다고 생각합니다.

하지만 이런 fluid design은 **무형성**과 **모호성**에 있어 어렵습니다. 그렇기에 일부 사례를 통해 시각화에서 fluidity를 정의하고, 이에 대한 가이드라인을 제시하며, 향후 연구 방향을 제시합니다.

## Fluid interaction for information viusalization

### 1. Operational definition

사전에서 fluidity는 다음과 같이 정의됩니다.

> **fluidity, n** : 1. the quality or state of being fluid; 2. the physical property of a substance that enables it to flow.

위의 사전적인 정의에서 우리는 정보 시각화에서 **_fluid interface_** 를 다음 특징들로 바라볼 수 있습니다.

#### Promotes flow

상호작용은 흐름 유지를 촉진하도록 설계해야합니다.

"Flow"는 참가자의 시도와 기술이 완벽하게 균형을 이루어 활동에 집중하고 참여하여 보람있는 결과로 이어지는 **활동에 완전히 몰입하는 정신 상태**로 정의됩니다.

미국 심리학자 미하이 칙센트미하이(Mihaly Csikszentmihalyi)이 말한 몰입 요인(factor of flow)을 통해 fluid interaction을 광범위하게 살펴봅시다.

> [몰입 flow 미치도록 행복한 나를 만난다](https://book.naver.com/bookdb/book_detail.nhn?bid=156269)

- **균형 잡힌 도전(Balanced challenge)** : activity가 요구하는 스킬과 사용자의 스킬 레벨의 매치
- **집중(Concentration)** : 제한된 관심 분야에 높은 수준의 집중
- **자의식 상실(Loss of self-consciousness)** : 사용자는 행동과 의식의 통합할 수 있어야 합니다.
- **시간의 변화(Transformation of time)** : _아 시간 녹았다_
- **신속한 피드백(Prompt feedback)** : 목표 달성에 대한 진행 상황을 즉시 피드백
- **통제 의식(Sense of Contorl)** : 사용자가 활동에 대한 통제를 느끼게 하여 결과와 직결되게
- **본질적 보상(Intrinsically rewarding)** : 활동 자체에 대한 실질적 보상

#### Supports direct manipulation

**직접 조작 패러다임(direct manipulation paradigm)** 은 도메인 객체 자체와 직접 상호작용하기 위해 명시적인 방법읍 사용하여 인터페이스의 간접성을 최소화합니다.

패러다임은 다음 네 가지 주요 원칙을 기반으로 합니다.

- 관심 객체를 지속적으로 표현 (Continuous representation of the object of interest)
- 복잡한 구문보다는 물리적 행동 (Physical actions instead of complex syntax)
- 신속하고 점진적이고 가역적 (Rapid, incremental, and reversible operations whose impact on the object of interest is immediately visible)
- 최소한의 지식으로 사용가능한 계층적 접근 (Layered or spiral approach to learning that permits usage with minimal knowledge)

#### Minimizes the gulfs of action

사용성(Usability) 전문가인 도널드 노먼(Donald A. Norman)에 따르면 물리적 또는 가상의 시스템과 상호작용하는 문제는 두 가지의 차이(gulf)로 설명할 수 있습니다.

> [도널드 노먼의 UX 디자인 특강](http://www.yes24.com/Cooperate/Naver/welcomeNaver.aspx?pageNo=1&goodsNo=59673763)

- **Gulf of Evaluation** : 시스템 상태와 해당 상태에 대한 사용자의 인식 차이
- **Gulf of Execution** : 시스템이 가능한 동작과 시스템 사용에 대한 사용자의 의도 차이

### 2. Towards a cognitive account of fluid interaction

> 위의 내용을 다수 줄글로 표현한 부분이라 생략합니다.

### 3. Utility of fluid interaction

유동성(fluidity)있는 상호작용이 꼭 정보 시각화에 필요한 것은 아닙니다.
이미 R등에서는 유동성 없이 효과적이고 성공적인 정보 시각화를 이뤘습니다.

허나 많은 저자들은 탐색을 위해 원활한 인터랙션의 필요성을 찬성합니다.
생산성에서 여러 문제가 있을 수 있으나 도구 혁신을 통한 분석 비용 구조 개선 등 이와 관련하여는 여러 이슈가 있습니다.

## InfoVis examplars

이제 예시들을 보며 좀 더 살펴보겠습니다.

총 6가지의 예시입니다.

![table](https://i.imgur.com/iW1Oomz.png)

### 1. Facet-Streams

**Facet-Streams([동영상](https://www.youtube.com/watch?v=giDF9lKhCLc))** 는 여러 사용자가 테이블 위에서 제품 검색을 하기위한 협업 시스템입니다.

제품의 카탈로그 등을 검색하여 의사 결정에 도움을 지웝합니다.
여러 토큰을 사용하여 AND나 OR 쿼리를 추가적 기호없이 사용할 수 있습니다.

터치 입력을 통해 일반적인 데스크톱 인터페이스보다 더 광범위한 사용자의 실제 운동 기술을 활용합니다.
시스템에 대한 모든 유형의 터치 입력은 즉각적인 시각적 피드백으로 이어지며 물리적인 상호작용에 도움을 줍니다.

초반에는 병렬적으로 각각 개별적인 작은 쿼리들을 네트워크를 구성하고, 필요에 따라 여러 사람과 네트워크를 구성할 수도 있습니다. 구성된 네트워크에서 필요한 부분을 제거하거나 추가하며 유연하게 작업을 진행할 수 있습니다.

요약하면 fluidity와 관련하여 3가지 포인트를 정리할 수 있습니다.

- 즉각적인 시각적 피드백이 이뤄지는 **실감적(tangible)인 터치 상호작용(touch interaction)**
- **낮은 점도(viscosity)** : 사용자의 개별 / 공유 목표에 따라 신속한 수정 가능
- **유연성(Flexibility in working styles)** : 사용자 간의 느슨하게 결합된 병렬 작업과 그에 따른 원활한 변경

![figure 1 & 2](https://i.imgur.com/krRCFCu.jpg)

### 2. BabyNameVoyager

**BabyNameVoyager** 는 과거 아기 이름에 대한 분석을 인터랙티브하게 보여주는 웹 기반 시각화 도구입니다. [link](https://www.babynamewizard.com/name-voyager#prefix=&sw=both&exact=false)

X축은 시간축, Y축은 이름에 따른 아기 수를 표현한 stack graph입니다.

사용자는 마우스를 통해 데이터를 탐색할 수 있습니다.
마우스가 가르키는 위치가 옮겨지면 원활한 상태 전환이 일어나며 정보간의 연속성을 제공합니다.

그리고 텍스트 쿼리를 사용하여 이름에 대한 필터링(filtering)을 할 수 있고, 필터링 규칙 및 내용에 따라 즉각적으로 시각적 표현이 업데이트 됩니다.

여기서는 fluidity와 관련하여 4가지 포인트를 정리할 수 있습니다.

- **자연스러운 애니메이션(Smooth animated transition)**을 통한 상태 전환
- **최소한의 인터페이스(Minimalistic interface)** : 클릭을 통한 직접적 조작 또는 텍스트 쿼리
- 마지막 상태뿐만 아닌 작업 중간 과정까지 **즉각적인 시각적 피드백(Immediated visual feedback)**
- 정확성을 유지한 미적 시각 디자인(Aesthetic visual design)

![figure 3](https://i.imgur.com/9YLlAG1.png)

### 3. ScatterDice and GraphDice

**ScatterDice** 과 **GraphDice** 는 다차원 테이프, 다변수 그래프를 탐색하기 위한 인터랙티브 시각화 툴입니다.

아래 이미지에서 볼 수 있듯이 2/3D 화면에서 산점도를 통해 데이터를 살펴볼 수 있으며 3D의 경우에는 회전 애니메이션 등을 통해 여러 시점에서 살펴볼 수 있습니다.

여기서는 fluidity와 관련하여 4가지 포인트를 정리할 수 있습니다. (BabyNameVoyager과 상위 3개 항목 동일)

- 사용자가 시각적 표현에 대해 생각하고 추론 할 수있는 **논리적인 개념 모델(Coherent conceptual model)**

![figure 4](https://i.imgur.com/LHL9jpo.png)

### 4. Mæve

Mæve 는 위의 Facet-Streams와 동작방식이 유사한 시스템으로 박물관 내의 다양한 건축 프로젝트 간의 관계를 나타내는 네트워크를 인터랙티브하게 조작할 수 있는 시스템입니다.

여기서는 단순히 터치 스크린 내에서만 상호작용하는 것이 아닌 '어떤' 결과를 보여줍니다. 단순히 결과가 아닌 실험으로의 확장으로 볼 수 있습니다.

예를 들어 영화와 관련된 특성에 관련된 결과는 어두운 공연 공간에 설치된 벽에 투영하여 결과를 보여주는 상호작용을 합니다.

기존의 결과가 터치 스크린에 국한되어 상호작용을 보여왔다면 Mæve는 유동성 개념이 시스템 스크린 미디어를 초월한 예시입니다. (터치 스크린 내부가 아닌 클래식 스크린 미디어의 일반적인 정보 시각화 응용 프로그램으로의 변환)

여기서는 fluidity와 관련하여 3가지 포인트를 정리할 수 있습니다.

- **응답가능(responsive)한 즉각적인(immediate)** 시각적 피드배
- 생산성보다 **경험에 초점**(Focus on experience rather than productivity)
- 실험을 장려하는 강하고 효과적인 입력과 출력

![figure 5](https://i.imgur.com/SCDIEAu.jpg)

### 5. We fill fine

**We fill fine** 은 현재의 사람 감정에 대한 인터랙티브한 탐색을 제공하는 서비스입니다. [link](http://www.wefeelfine.org/)

색, 크기, 모양, 불투명도를 이용한 특성 표현과 픽토그램 또는 여러 축으로 구성된 시각적 패턴이 특징입니다.

다양한 패턴으로 사람의 감정을 표현하며 매력적이며, 다른 목표가 아닌 본능적인 재미로 상호작용을 진행하게 됩니다. 즐거움이라는 보상이 있는 경험이지만 실제 사용자는 상호작용을 통해 경험을 얻습니다.

미학뿐만 아니라 재미, 내용, 유연성 등을 모두 제공하는 서비스라는 것에 의미가 있습니다.

여기서는 fluidity와 관련하여 3가지 포인트를 정리할 수 있습니다.

- **즉각적이고 재미있는(Immediate and playful)** 상호작용
- 탐구를 장려하여 **끝나지 않는 상호작용(Never-ending interaction flow)**
- 서로를 보충하는 **전체적인 상호작용, 시각 디자인, 내용(Holistic interaction, visual design, and content)**

![figure 6](https://i.imgur.com/hrqwcWW.png)

### 6. Interactive Holographics in Iron Man 2

컴퓨터 인터페이스에 대한 인기는 영화와 미디어에서도 많이 활용되었습니다.
대표적으로 아이언맨이 작업할 때 사용하는 디스플레이 및 홀로그램이 그 예시입니다.

특히 특수 효과 디자인 회사 Prologue에서 디자인한 조밀한 정보 대시보드와 반응성이 뛰어난 실시간 3D 인터페이스는 미래를 상상하게 만듭니다.
손짓, 손가락, 스냅, 핀치로 정보를 즉시 쿼리하고 결과를 옆으로 밀고 던지는 등 일상적인 작업에서 수행의 재미를 만드는 비전을 제시합니다.

- 제스처 및 바디랭귀지 등 자연스러운 상호작용을 데이터 및 시각적 디스플레이와 상호작용하는 **현실 기반 상호 작용(Reality-based interaction)**
- 즉각적인 시각적 피드백

![figure 7](https://i.imgur.com/1RgacYR.jpg)

## Design guidelines for fluidity

- DG1 : 상태를 변화시킬 때, 부드러운 애니메이션을 사용하라 (Use smooth animated transitions between states.)
- DG2 : 상호작용에 대한 즉각적인 시각적 피드백을 제공하라 (Provide immediate visual feedback on interaction.)
- DG3 : 인터페이스에서 간접성을 최소화하라 (Minimize indirection in the interface.)
- DG4 : 시각적 표현에 사용자 인터페이스 구성 요소를 통합하라 (Integrate user interface components in the visual representation.)
- DG5 : 상호작용에 대한 보상 (Reward interaction.)
- DG6 : 상호 작용이 '끝'나지 않도록 하라 (Ensure that interaction never ‘ends.’)
- DG7 : 명확한 개념적 모델을 강화하라 (Reinforce a clear conceptual model.)
- DG8 : 명시적인 모드 변경을 피하라 (Avoid explicit mode changes.)

## Vision and research directions

### InfoVis Exemplear Repository

Infovis의 예제를 수집하여 저장소를 만드는 것이 첫 번째 연구 방향입니다.

이러한 저장소는 이 논문처럼 큐레이팅되어야 하며 각 예시의 **장점과 약점** 을 강조하기 위해 제작자 외의 **다양한 의견** 을 모두 포함해야 합니다.

좀 과거의 자료이긴 하지만 논문에서 언급한 자료의 링크는 다음과 같습니다. (링크가 없는 사이트는 다 사라졌...)

- information aesthetics]
- [Manuel Lima’s VisualComplexity.com](http://www.visualcomplexity.com/vc/)
- Michael Friendly’s Gallery of Data Visualization
- [Robert Kosara’s EagerEyes](https://eagereyes.org/)
- Potsdam Information Design Patterns website

### Visualization Design Patterns

디자인 패턴을 공식화 하는 것이 두 번째 연구 방향입니다.

- 용어 표준화
- 모범 사례 영속화
- 시각화 품질 개선
- 안좋은 패턴 정의
  - rainbow scale
  - 심각하게 산만한 애니메이션
  - 지나치게 복잡한 시각화

### Towards Visualization Criticism

세 번째는 디자인 분야의 교육과 전문적 실습의 통합화입니다.

- 시각적 표현과 상호작용의 동등한 취급
