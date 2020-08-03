---
title: "[HCI] Toward a Deeper Understanding of the Role of Interaction in Information Visualization 리뷰"
category:
  - hci
tag:
  - interactive
  - data visualization
  - paper review
author_profile: true
header:
  overlay_image: https://images.unsplash.com/photo-1477951233099-d2c5fbd878ee?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2250&q=80
  overlay_filter: 0.7
---

InfoViz에서 Interaction의 종류 7가지

[Toward a Deeper Understanding of the Role of Interaction in Information Visualization(5007)](https://www.cc.gatech.edu/~stasko/papers/infovis07-interaction.pdf) 논문에서 언급한 Information Visualization에서 7가지 Interaction에 대해 가볍게 정리해보았습니다.

- **Select**: mark something as interesting
- **Explore**: show me something else
- **Reconfigure**: show me a different arrangement
- **Encode**: show me a different representation
- **Abstract/Elaborate**: show me more or less detail
- **Filter**: show me something conditionally
- **Connect**: show me related items

## 1. Select : mark somting as interesting

![figure 1](https://i.imgur.com/rgokBCw.png){:width="500px"}

**_Select_** 는 관심이 있는 데이터(아이템)을 표시하여 추적하는 기법입니다.
한 번에 파악하기 어려운 많은 데이터에서 보고 싶은 데이터를 쉽게 추적할 수 있습니다.

말장난 같지만 이런 *Select*는 후속 작업에 대한 선행 작업으로 사용합니다.

재정렬을 하기 전 미리 데이터를 선택하고, 정렬 후에 새로운 위치를 확인하는 것처럼 말입니다.

독립적으로 사용하기 보다는 다른 interaction들과 결합되어 사용자의 exploratin과 discovery를 돕습니다.

## 2. Explore : show me somthing else

**_Explore_** 는 데이터의 부분집합을 살펴보는 기법입니다.

흔히 정보 시각화에서 데이터가 너무 많은 경우에는 한정된 수의 데이터만 볼 수 있습니다. (컴퓨터의 한계, 인지능력의 한계 등)

그래서 *Explore*를 통해 데이터의 하위 부분 집합을 확인하고, 다음 다른 데이터들을 보려고 진행합니다.

하지만 꼭 보고있는 view가 완전히 변경되는 것은 아닙니다. 보통 일부 새로운 항목이 추가되고, 기존 일부 항목이 제거되는 과정이 반복됩니다.

가장 일반적인 Explore는 **Panning**입니다.

Panning이란 카메라가 정지 된 상태에서 장면 또는 장면 이동을 통한 카메라의 움직임을 말합니다.

종종 마우스로 이동또는 스크롤 막대를 통해 화면 전환을 하는 것도 포함합니다.

> 프레지(Prezi)를 떠올리면 쉬울듯 합니다. 하 이것도 요새 애들은 모르겠죠 :(

또 다른 Explore 예시는 **Direct-Walk** 입니다.
이는 정보 구조에서 보는 초점의 위치를 바로 변경하는 것을 의미합니다.
하이퍼링크(Hyperlink)가 이에 대한 예시입니다.

> 이건 개인적으로 미니맵 등을 떠올리면 쉬울 것 같습니다.

![figure 2](https://i.imgur.com/9FKCouR.png){:width="500px"}

## 3. Reconfigure : show me a different arrangement

**_Reconfigure_** 는 표현의 정렬을 달리하여 다른 관점으로 볼 수 있는 기법입니다.

정보 시각화의 주요 목표는 데이터의 숨겨진 특징과 그 사이의 관계를 알아내는 것입니다.
하지만 단 한 개의 표현으로는 그런 내용을 확인하기 어렵습니다.

그렇기에 Reconfigure는 데이터를 여러가지 방법으로 정렬하며 특징을 찾기 위해 노력합니다.

![figure 3 & 4](https://i.imgur.com/ZQC4RYO.png){:width="500px"}

이런 Reconfigure는 데이터가 많아지면서 발생하는 **중첩** 문제를 해결하는 데도 사용할 수 있습니다.

- 3D roatation
- Jitter (이 개념을 명확하게 설명해주실 분 구합니다ㅠ)

![figure 5](https://i.imgur.com/yVzpmS2.png){:width="500px"}

## 4. Encode : show me a different representation

**_Encode_** 는 각 데이터 요소의 시각적 모양(색, 크기, 모양)을 포함한 데이터의 기본 시각적 표현을 변경하는 기법입니다,

기본적으로 정보 시각화는 시각적 요소에 대한 선행 지식이 바탕이 되기 때문에 인지적인 요소에 직결되고 그만큼 이해를 도울 수 있습니다.

이에 대한 예시는 다음과 같습니다.

- 고도 표시를 색을 통해 Contour로 나타내면 식별이 쉬움
- Pie Chart를 Histogram으로 변경
- 색상 및 색상 스펙트럼 조정

## 5. Abstract/Elaborate : show me more or less detail

**_Abstract/Elaborate_** 는 data representation의 abstraction 수준을 조정하는 기법입니다. 즉, 정보를 얼마나 더 대충/세밀하게 볼까의 조정입니다.

- Interactive Treemap (_논문에서는 SequoiaView, Cushion Tree라고 하는데 요새는 Treemap으로 퉁치는거 같다._)
- SubBurst
- 마우스 커서를 올렸을 때 정보를 제공하는 **tooltip** 도 이에 대한 예시입니다.
- **zooming**도 한가지의 예시로, 표현의 배율을 높이거나 줄여 데이터셋을 크게/작게 볼 수 있습니다.
  - _Explore와 유사한 면이 있습니다._

결론적으로는 정보를 얼마나 제공할까의 인터랙티브입니다!

![figure 6](https://i.imgur.com/4iAeZwP.png){:width="500px"}

## 6. Filter : show me somthing conditionally

**_Filter_** 는 특정 컨디션에 따라 표현하는 데이터 셋을 조정하는 기법입니다.

조건이나 범위에 따라 해당 항목이 충족되는 데이터만 시각화에 사용됩니다.
실제 데이터는 변하지 않지만 디스플레이에서는 내용이 숨겨지거나 다르게 표현됩니다.

- 동적 쿼리 컨트롤 (Dynamic query control)
  - 슬라이더 또는 특정 값을 이동하여 범위 선택하면 해당 내용을 즉시 보여줌
  - alphasliders, rangesliders, toggle button 등을 사용하여 텍스트, 숫자, 범주 데이터를 필터링합니다.
- Attribute Explorer(Fig 7)
  - 필터링된 내용을 디스플레이에서 제거하지 않고, 색상을 변경하여 동적 쿼리 기능을 확장합니다.
- Fig 8과 같이 keyboard interaction도 가능합니다. (이름 필터링 like 검색 시스템)
- QuerySketch

![figure 7 & 8](https://i.imgur.com/Dwxbfz8.png){:width="500px"}

## 7. Connect : show me related items

**_Connect_** 는 (1) 이미 표시된 데이터 항목 간의 연관 및 관계를 강조하고, (2) 지정된 항목과 관련된 숨겨진 데이터 항목을 표시하는 기법입니다.

- Fig 9는 같은 데이터 다른 표현에서 좌측에 mark한 데이터를 우측 표현에서도 mark해준 예시다.

![figure 9](https://i.imgur.com/lrRQdfv.png){:width="500px"}

## 8. ETC techniques

- **_Undo/Redo_** : Ctrl + Z 와 Ctrl + Shift + Z
- **_Chnage configuration_** : 다양한 설정 변경

## 느낀점

- 확실히 구분하기는 어려울 수 있으나, 이런 분류가 있으니 잘 사용하면 될 것 같다는 개인적인 결론
- 추후 논문도 찾아보고 리뷰할 예정입니다.
