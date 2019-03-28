---
title : Jupyter로 시작하는 스터디 생활
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
  teaser : /assets/images/category/ml.jpg
  overlay_image : /assets/images/category/ml.jpg
published : false
---

스터디를 위한 준비 jupyter notebook에서 해봅시다.


> 본 포스트의 공유는 좋지만, 출처는 꼭 밝혀주시면 감사하겠습니다. :-)

## Introduction

Python, Kaggle, Keras, Tensorflow 커뮤니티를 보면 정말 많은 스터디가 존재합니다.
저도 스터디 및 강의 하는 것을 좋아하고, 이번 학기에는 머신러닝/캐글 스터디장을 하나 하게되었습니다.

해보신 분들은 알지만 블로깅/영상촬영을 하면 가장 힘든 것은 공부와 강의가 아니라 **자료** 만들기 입니다.

스터디를 위한, 공유를 위한, 발표를 위한 **Jupyter로 시작하는 스터디 생활** 시작합니다.

> 본 포스팅은 맥북 유저가 작성한 포스팅입니다. 윈도우 설치 등에 있어 부족한 내용이 있을 수 있습니다. 부족한 부분에 대해 요청 또는 피드백을 해주시면 추가하겠습니다.

실습 영상은 Python으로 진행했습니다.

## I. Jupyter Notebook

### I-1. Jupyter notebook 이란?

Jupyter Notebook이란 IPython을 바탕으로 만든 오픈 소스 웹 응용 프로그램입니다. 캐글과 구글 Colab등에서 사용하기 때문에 알아두면 데이터 분석을 공부/연구하는 분이라면 필수적인 도구 중 하나입니다.

공식 사이트에서 [제공하는 데모](https://jupyter.org/try)를 사용해볼 수 있습니다.
물론 오픈소스이기에 설치는 무료입니다.

#### IPython

> Jupyter는 IPython을 기반으로 합니다.

**IPython** 은 Interactive Python에서 나온 명칭으로 쉽게 Python + Shell입니다.

Python의 Interactive 한 성질과 shell 명령어를 사용할 수 있다는 장점이 있습니다.
이제 이런 모습을 웹으로 올린 것이 Jupyter Notebook이라고 생각하면 됩니다.

<figure>
 <img src = "https://i.imgur.com/XOFfb8V.png" >
 <figcaption> 수비니움의 코딩일지 많이 사랑해주세요. 좋아요와 구독도... </figcaption>
</figure>

#### 대화형 컴퓨팅

IPython과 같이 Jupyter는 command shell을 이용해 계속 대화형으로 코딩할 수 있습니다.
이는 데이터 분석에 있어서 필요한 부분만 반복해서 실행할 수 있다는 장점이 있습니다.

#### 지원 언어

총 40개의 언어를 지원하고, 그 중에서도 머신러닝에서 많이 사용하는 Python, R, Julia, Scala를 지원합니다.

#### 공유 기능

IPython notebook 파일을 쉽게 공유하기 위해 [Jupyter nbviewer](https://nbviewer.jupyter.org/) 기능을 제공하고 있습니다.

#### Markdown 작성 가능

제가 가장 좋아하는 기능 중 하나는 Markdown을 통해 문서를 작성할 수 있다는 점입니다.
마크다운 작성은 문서를 더욱 쉽게 편집할 수 있고, 후에 만들 Slide 나 포스팅에 유용합니다.

#### Jupyter Notebook & Jupyterhub

저는 Jupyterhub은 사용하지 않으니 이 부분은 생략합니다.
필요한 분이 있다면 추후에 적어보겠습니다. (TBD)

### I-2. Install

### Windows

Windows의 경우에는 anaconda를 사용하면 좋습니다.

윈도우 사용자 분들은 jupyter notebook 을 *1. 명령 프롬프트에 치거나* 또는 *2. 검색해서 클릭하면 됩니다.*

### MacOS

MacOS 또는 리눅스라면 pip로 설치합시다. 터미널에서 다음과 같이 실행하면 됩니다.

``` shell
# Python3을 설치합시다.
brew install python3

# Python3 version이므로 pip3 사용
pip3 install jupyter notebook
```

Terminal에서 다음과 같은 명령어를 이용해 열 수 있습니다.

``` shell
jupyter notebook
```

모두 local 서버에서 열립니다.

### I-3. Interface

### I-4. 세부 기능

### I-4. 단축키

> 여기서부터 실습 영상을 제공합니다.

[수비니움의 IPython 사용하기]()

## II. Theme Setting

기본 테마도 마음에 들지만, Custom을 좋아하시는 분들이라면 필수입니다.

테마에 따라 가독성이 높아질 수 있고, 훨씬 더 높은 퀄리티의 자료를 만들 수 있습니다.

## III. Jupyter Slides

## Conclusion

## Reference

### Official

- [Jupyter Official Page](https://jupyter.org/)

### blog/slide

- o블링블링o님 : [Jupyter의 기본개념](https://yhzion.tistory.com/15)

- 변성윤님(@zzsza) : 10분만에 익히는 Jupyter Notebook [Slide Share](https://www.slideshare.net/zzsza/10-jupyter-notebook)

### book

- [파이썬 라이브러리를 활용한 데이터분석]()
