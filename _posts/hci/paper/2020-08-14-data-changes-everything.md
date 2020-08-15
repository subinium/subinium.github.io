---
title: "[HCI] Data Changes Everything : Challenges and Opportunities in Data Visualization Design Handoff 리뷰"
category:
  - hci
tag:
  - interactive
  - data visualization
  - paper review
author_profile: true
header:
  overlay_image: https://images.unsplash.com/photo-1581291518857-4e27b48ff24e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80
  overlay_filter: 0.7
---

Data Visualization Process에서 생기는 여러가지 문제점

데이터 시각화 프로젝트 대규모팀이 시각화를 진행하며 생길 수 있는 문제점과 그 문제점에서 얻을 수 있는 여러 기회에 대한 리뷰 형식의 논문입니다.

전체적으로 데이터 시각화 분야에서 디자인과 개발의 프로세스에서 생기는 문제점들을 고려할 때, 읽어보면 좋은 논문인 것 같습니다.

IEEE VIS 2019 Best Papers 중 하나라 읽어봤는데, 상당히 흥미롭고 재미있네요.

- [Data Changes Everything : Challenges and Opportunities in Data Visualization Design Handoff](https://arxiv.org/abs/1908.00192)

## Abstract

복잡한 데이터 시각화 프로젝트에는 시각화 관련 스킬이 다른 여러 사람의 협업이 수반됩니다.
예를 들어 시각화 디자인을 담당하는 디자이너와 그 결과를 구현하는 개발자가 모두 포함됩니다.

우리는 데이터 특성화 도구(data characterization tools), 시각화 디자인 도구, 개발 플랫폼 등 디자이너와 개발자간의 시각화 협업에 있어 차이를 확인합니다.

그 과정에서 5개의 대규모 다분야 시각화 설계 프로제트에서의 관찰 및 반성을 통해 디자인 명세 및 핸드오프를 위한 6가지 데이터 시각화 챌린지를 보고합니다.

이러한 관찰을 기반으로 성공적이고 협업이 가능한 데이터 기반 설계의 프로토타이핑, 테스트 및 전달 등 향후 데이터 시각화 도구 개발에 있어 기회를 제시합니다.

## Overview of Visualization Design Projects

![figure 2](https://i.imgur.com/8osydQU.png)

- Projects (프로젝트 종류와 링크)
  - Energy Visualization for the Inter-American Development Bank
(IDB)
  - [Energy Future](http://apps2.cer-rec.gc.ca/dvs/?page=landingPage&language=en)
  - [Pipeline Incidents](http://apps2.cer-rec.gc.ca/pipeline-incidents/)
  - [Energy Imports & Exports](http://apps2.cer-rec.gc.ca/imports-exports/)
  - Pipeline Conditions
- Design Team Roles
  - characterize data
  - create and understand data mapping
  - visual design
  - interaction design
  - developed visualization prototypes
  - engineered
  - collaborate & communicate
- Analysis and Synthesis Process

## Visualization Design & Development

![figure 3](https://i.imgur.com/FdGoQsC.png)

데이터 시각화의 절차는 크게 다음 단계를 거친다.

- **Project Conceptualization** : 클라이언트 측에서 디자이너에게 비전과 목표를 제시하는 단계
- **Data Characterization** : 데이터 탐색 및 분석, 추후 시각화 작업에 필요한 데이터 파악
- **Visualization Design** : 데이터에 따른 디자인 작업 
  - 과정 내의 data mapping 과정은 초안을 일러스트레이트 툴로 만드는 과정을 의미
  - 2가지 절차
    1. 시각화 컨셉을 만들고, 클라이언트 초기 승인
    2. 수정 및 추가 작업을 통한 최종 디자인과 문서화 (visualziation design documentation) 
- **Visualization Development** : 디자인 내용을 바탕으로 구현
- **Deployment and Use** : 배포

## Challenges when Designing with Data

- **C1. Adapting to Data Changes**
  - 시각화 개발은 데이터에 의존하기에 데이터가 업데이트되면 개발 단계 및 시각화 결과 등에서 여러 현상이 발생할 수 있다.
- **C2. Anticipating Edge Cases**
  - 디자이너는 모든 인터랙션의 조합들을 테스트하고 예측하기 어렵고, 그런 이유로 종종 에러가 발생할 수 있다.
- **C3. Understanding Technical Constraints**
  - 디자이너는 모든 개발적인 한계를 신경쓸 수 없다. 그렇기에 개발 단게에서 큰 개선이 필요한 경우가 생길 수 있다.
- **C4. Articulating Data-Dependent Interaction**
  - 데이터 의존적 상호작용의 명료화, 새로운 데이터 인터랙션을 추가할 때 정확한 표현을 하기 위해서 수 많은 여러 상태의 뷰를 만들며 시행착오가 있을 수 있으며 이 과정에서 extra cost가 발생하고 팀과 개발에 있어 소통이 어려울 수 있다.
- **C5. Communicating Data Mappings**
  - data mapping은 mockup에 비해 구현은 더 많은 디테일과 정확도를 필요로 한다. 하지만 design tool에서 data-mapping의 기능을 완전히 '잘' 지원해주지 않는다.
- **C6. Preserving Data Mapping Integrity across Iterations**
  - 구현과 설계 문서를 체계적으로 비교하는 것은 어렵습니다. 그렇기에 개발 과정 또는 이후에 data mapping의 오용 또는 오해 등의 문제가 발생할 수 있습니다.

## Discussion

위에서 말한 challenge들은 디자이너와 이를 구현하기 위한 개발자의 본질적인 연결을 위한 문제들입니다. 
이 challenge를 통해 개인 및 협업 팁 모두를 위한 시각화 디자인 프로세스를 구체적으로 지원할 수 있는 연구 및 도구 생성을 위한 논의점을 제시합니다.

### Data Characterization

**C1** 의 문제를 해결하기 위해 다음과 같은 부분을 논의할 수 있습니다.

- 데이터 의존 시각화의 견고성을 위한 명확한 프로세스가 필요합니다.
- 데이터 특성화 도구는 데이터의 변경사항이 시각화 디자인에 미치는 영향을 이해시켜 디자이너를 돕는다.
- 이를 위해 데이터 열의 이름, 극값 및 통계 분포의 변경 사항을 강조 표시, 현재 분포를 기반으로 변경되는 상황을 시뮬레이트 하는 등의 내용을 포함할 수 있습니다.
- 최근 `Orange`나 `DataTours` 등의 도구의 이상값 감지 및 데이터 마이닝 프로파일링 등의 반자동 방식의 접근법은 이에 대한 좋은 시도입니다.
- 서로 다른 데이터셋의 분포를 빠르게 비교하기 위한 시각적 도구는 디자이너가 통계에 의존하지 않고 문제가 있는 변경 사항을 보다 쉽게 감지하는데 도움이 될 수 있습니다.

### Design Phase

- **Data-Driven Visualization Ideation**
  - 우리가 경험한 다수의 문제는 데이터 기반 시각화의 구상과 관련되어 있습니다.
  - `Adobe Illustrator` 등 수동적인 벡터 기반 그래픽 디자인 툴은 복잡한 데이터 기반 뷰 생성을 한정적으로 지원하고,
  - `Tableau` 또는 `RAWGraphs` 등의 시각화 탐색 및 생성도구는 한정적으로 커스텀 시각화 및 인터랙션을 지원합니다.
  - 하지만 최근의 `Data Illustrator`, `Data Ink`, `Data-Driven Guides`와 같은 도구는 표현력있는 데이터 중심 그래픽 디자인 도구의 잠재력을 강조합니다.
  - 직접적이고 동적이고 표현적인 도구는 **C1** 문제에 다양한 설계 대안을 빠르게 탐색 가능
  - 빠른 탐색으로 **C2**. **C4** 도 빠르게 발견
  - 이를 위해 `Hanpuku`와 같은 도구는 `Adobe Illustrator`의 그래픽 디자인 표현력과 `D3`의 데이터 기반 프로토타이핑 기능을 연결하는 방법을 탐색했으나 아직 부족합니다.
- **Data-Driven Interaction Prototyping**
  - 시각화 디자인 내에서 데이터 기반 상호작용을 프로토타이핑하는 것은 상호작용 옵션의 확장성과 이해도를 높이며 개발자에게 전달하는데 중요합니다. (**C4**)
  - 그러나 현재는 레이아웃, 모양 등 데이터에 의해 좌우되는 시각화 인터랙션을 정적으로 간단하게 프로토타입을 만들며 표현력이 부족한 부분이 많습니다.
- **Data Mapping Documentation**
  - 설계 의도를 전달하고, 이를 문서화하는 작업은 개발 및 소통에 있어 유용합니다.
  - 필요한 모든 변환, 계산 및 알고리즘 등의 세부 사항을 사용하여 데이터 구조와 그래픽 표현 간의 관계에 대한 명시적 커뮤니케이션을 지원합니다. (**C5**)
  - 다음과 같은 연구들이 시도되고 있습니다. (*다 읽어볼려고 논문 제목까지 긁어왔습니다.*)
    - 시각화 문법 
      - Vega-Lite: A Grammar of Interactive Graphics 
      - Polaris: A system for query analysis and visualization of multi-dimensional relational databases
      - Thought: A Trail Map for the 21st Century
    - 시각화 파이프라인 
      - Lark: Coordinating Colocated Collaboration with Information Visualization
    - 시각화 데이터 매핑 해체 및 수정 
      - Deconstructing and Restyling D3 Visualizations
- **Data Visualization Design Documentation**
  - 데이터 매핑은 시각화의 기본이지만 디자인의 일부분입니다. 레이아웃, 타이포그래피, 색상 등과 상호작용을 위해서는 더 큰 디자인 문서가 필요합니다.
  - Bret Victor가 대중화한 형식은 데이터 분석 실무에서 점점 더 많이 사용되고 있으며, `Observable`, `litvis`, `Idyll` 둥의 툴은 이런 시각화 디자인 문서화를 제공할 수 있습니다.
  - 주석 도구도 중요한 역할을 하며 `cLuster`, `D.Note`, `SketchComm`, `SILK` 등이 이를 위한 기능을 제공합니다. `ChartAccent`와 같은 시각화 관련 주석 라이브러리도 있습니다.

### Development Phase 

**C6** 에서 강조한 바와 같이 구현된 시각화를 설계 문서에 대해 테스트하는 사람은 누구나 설계와 구현 간의 불일치를 식별할 수 있어야합니다.

가장 중요한 불일치는 데이터 매핑(**C5**)입니다.
시각화 간의 차이와 시각적 비교를 지원하는 다음과 같은 도구가 유용할 수있습니다.

- Considerations for Visualizing Comparison
- Visual comparison for information visualization

최종 상호작용을 시뮬레이션하는 작업과 프로토타입은 이러한 문제를 일부 완화할 수 있으나 이는 디자인팀과 설계팀이 **C3** 와 같은 기술적 제약에 대한 논의가 미리 이루어진 경우에 가능합니다.
