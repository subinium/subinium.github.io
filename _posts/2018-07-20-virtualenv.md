---
title :  vitualenv를 써봅시다
category :
  - development
tag :
  - virtualenv
  - python
  - library
sidebar_main : true
header:
  teaser : /assets/images/category/devel.jpg
  overlay_image : /assets/images/category/devel.jpg
---

virtualenv 조금은 알고 쓰자!

## virtualenv를 포스팅하게 된 계기

딥러닝을 공부하기 위해 matplotlib 라이브러리를 사용해서 그래프 예시를 그려보려 했으나 이상한 에러 코드가 나서 실패했다.
<font size="1em"> *(이 문제는 DB과제를 하며 그래프를 그릴때도 똑같은 에러가 발생했는데, 그 당시에는 에러해결보다 png파일로 저장하여 이미지를 GUI에서 불러오는 방식을 선택했다.)* </font>

그렇게 찾아본 결과, virtualenv에서 shell rc코드를 변경하고, 뭐시기뭐시기 하라는데 자꾸 virtualenv를 모르고 쓰니 설정을 변경하기 무섭다.

그렇기 때문에 이번 기회에 어떤식으로 virtualenv를 활용할 수 있는지 공부해본다.

## 어떤 상황에 필요한 걸까?

python으로 과제를 처음에 했을때는 파이썬을 terminal에서 실행하면 다음과 같은 코드로 실행한다.

```shell
python test.py
```

과제를 처음해보는 사람 *(ex. subinium)* 이라면 루트에서 python을 실행하는 경우가 대부분이다. 이렇게 한다고 해서 문제가 생기는 것은 아니다. 하지만 다음과 같은 상황은 어떨까?

> 수빈이는 기말 대체 과제가 3개 밀렸다. 하지만 교수님 세 분이 공지한 채점용 개발환경 python 버전이 모두 다르다. 이때 수빈이가 할 수 있는 최선의 선택은 무엇일까?

이 상황에서 문제 해결은 다음과 같은 케이스로 나누어질 수 있다.

0. naive solution
    - 각 과제를 실행할때마다 root환경을 바꿔주며 실행
1. 부자 solution
    - 컴퓨터를 3대 산다.
    - ~~대리과제 등등~~
2. 협업 solution
    - 3인 이상 과제 공동체를 형성한다.
3. 컴퓨터를 조금 배운 solution
    - virtual box에서 OS를 3개 설치해서 각각의 환경을 다르게 한다.
4. <font color = "red">smart solution</font>
    - 한 컴퓨터 내에 가상환경 3개를 만들어, PATH를 따로 설정하여, 루트 설정을 필요에 따라 환경을 추가하거나 변경하여 빠르고 쉽게 해결한다.

4와 같은 획기적인 생각을 이미 선대의 똑똑한 개발자분들이 만들어 놓은것이 virtualenv다. 그럼 다음 장에서 제대로 알아보도록 하자.

## virtualenv 사용법

[원문](https://virtualenv.pypa.io/en/stable/#)

위에서 설명한대로 virtualenv는 Python 개발 환경을 나누는 도구다.

사용하기 위한 방법을 5개로 나누어 보면 다음과 같다.

### Step1. 설치

일단 프로그램을 설치해야 실행할 수 있으니 설치부터 하자.

pip명령어가 있다면 쉽게 install할 수 있다.

```shell
$ pip install virtualenv
```

설치 끝. 참고로 이 라이브러리는 가상환경 내에 설치할 수는 없다.

### Step2. 파일 설정

```shell
$ mkdir [project directory]
$ cd [project directory]
$ virtualenv [enviroment_name]
```

이렇게 되면 가상환경이 만들어졌다.
보통 enviroment_name은 *venv, env, .env, .venv* 등으로 설정한다. 다음 스텝부터는 **venv** 로 예시 코드를 작성한다.

### Step3. 가상환경 활성화

```shell
$ . venv/bin/activate
```
이 코드를 실행하면, 이후부터는 어느 디렉토리에 있더라도 파이썬 환경의 기준은 이 디렉토리이다.

항상 명심하며 step6을 보도록하자.

### Step4. 라이브러리 설치 및 개발

pip install 로 원하는 라이브러리 설치하고 개발하면 된다.

### Step5. 비활성화

step3에서 말한대로 활성화하면 파이썬 환경의 기준은 한 디렉토리다. 그렇기에 다른 환경을 개발하기 위해서는 deactivate를 해주어야한다.

```shell
$ deactivate
```

완료.

## 앞으로 포스팅할 내용

1. virtualenv에서 PATH 설정법
2. pyenv, autoenv를 통한 개발환경 구축

#### Reference
<a href = "http://ulismoon.tistory.com/3?category=553187">[Python] 독립적인 가상의 파이썬 실행환경, virtualenv (1)(2)</a> @ulismoon
