---
title : \[Keras Study\] Mac에서 Keras 환경 구축하기
category :
  - ML
tag :
  - keras
  - macos
  - deep-learning
  - AI
  - tool
  - python
sidebar:
  nav: sidebar-keras
use_math : true
header:
  teaser : /assets/images/category/devel.jpg
  overlay_image : /assets/images/category/devel.jpg

---
이게 과제라니.. 이게 나라냐

## Why Keras?

일단 [인공지능]수업 **과제** 중 하나로 나왔습니다...!

기말고사가 다음주인데 term project성 final 대체과제라니 충공깽 그자체;;

Keras가 무엇이고 왜 중요한가는 다음에 포스팅하고, 우선 빠르게 Keras를 진행해보려고 합니다.

머신러닝 이론 정리 및 keras 관련 자세한 내용보단 역시 실습을 바로 진행하는 게 재밌네요.


## Keras 환경 구축하기

### 환경

본 환경은 macOS Mojave 10.14.2 18C54 x86_64에서 진행되었습니다.

이 글은 zsh에서 homebrew, pip, vim이 설치되어 있다고 가정하고 작성되었습니다.

일부 라이브러리 등은 최근 일부 python version에서 진행되지 않을 수 있으니 꼭 체크해주셔야 합니다.

tensorflow가 3.7 이상의 버전이 없어 3.6.7에서 프로젝트를 진행하였습니다.


### 디렉토리 / 가상 개발환경 만들기

본 프로젝트는 `Keras`라는 디렉토리에서 진행될 예정입니다.

```
$ mkdir Keras
$ cd Keras
```

virtualenv을 이용하여 가상환경을 만들어줍니다.

설치가 안된분은 아래의 명령어로 설치해주시면 됩니다.

```
$ pip install virtualenv
$ virtualenv venv
```

이후 가상환경의 실행은 다음과 같습니다.

```
$ source venv/bin/activate
```

실행이 성공했다면 (venv)라는 문구를 확인할 수 있습니다.
가상환경 종료는 다음과 같은 명령어로 가능합니다.

```
$ deactive
```

### 주요 패키지 설치

다음은 keras에서 진행하기 위한 파이썬 패키지들입니다.

```
$ pip install numpy
$ pip install scipy
$ pip install scikit-learn
$ pip install matplotlib
$ pip install pandas
$ pip install pydot
$ pip install h5py

// 그래프 가시화를 위한 툴
$ brew install graphviz
```

### GPU 설정하기

GPU가 꼭 필요하지는 않지만 코드의 속도에는 필요한 요소라고 합니다.
일단 필수가 아니기에 후에 업데이트하겠습니다. (그리고 이것은 잘못된 생각이었다. 미리할걸)

### 딥러닝 라이브러리 설치

텐서플로우를 설치하면 씨아노를 설치할 필요는 없습니다.
하지만 가끔 케라스 모델을 만들 때, 서로 바꾸어 사용하는 것이 유용할 수 있다고 합니다.

모두 설치해봅시다.

```
pip install theano
pip install tensorflow
pip install keras
```

### 케라스 예제 실행해보기

위와 같이 pip로 설치할 수 있지만 git을 통해 다운받을 수도 있다.
이 경우 keras의 수 많은 예제를 진행할 수 있다.

```
$ git clone https://github.com/keras-team/keras
$ cd keras
$ python setup.py install
```

mnist예제도 있으니 진행해보면 다음과 같다.

```
$ python examples/mnist_cnn.py
```

아래는 mnist_cnn을 진행해본 예시이다. 다른 툴 없이 노트북에서 진행한 결과 20분정도 소모되었다.

<figure class = "align-center" style = "width : 580px">
  <img src= "/assets/images/keras/keras_example.png" width="580" alt>
  <figcaption> terminal창 스크린샷 </figcaption>
</figure>

```
venv ❯ python examples/mnist_cnn.py
Using TensorFlow backend.
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
11493376/11490434 [==============================] - 42s 4us/step
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
2018-12-13 23:44:42.476460: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
60000/60000 [==============================] - 78s 1ms/step - loss: 0.2596 - acc: 0.9189 - val_loss: 0.0548 - val_acc: 0.9829
Epoch 2/12
60000/60000 [==============================] - 82s 1ms/step - loss: 0.0877 - acc: 0.9735 - val_loss: 0.0462 - val_acc: 0.9843
Epoch 3/12
60000/60000 [==============================] - 77s 1ms/step - loss: 0.0649 - acc: 0.9807 - val_loss: 0.0331 - val_acc: 0.9878
Epoch 4/12
60000/60000 [==============================] - 77s 1ms/step - loss: 0.0524 - acc: 0.9838 - val_loss: 0.0304 - val_acc: 0.9894
Epoch 5/12
60000/60000 [==============================] - 77s 1ms/step - loss: 0.0463 - acc: 0.9861 - val_loss: 0.0288 - val_acc: 0.9902
Epoch 6/12
60000/60000 [==============================] - 78s 1ms/step - loss: 0.0429 - acc: 0.9870 - val_loss: 0.0273 - val_acc: 0.9901
Epoch 7/12
60000/60000 [==============================] - 78s 1ms/step - loss: 0.0377 - acc: 0.9886 - val_loss: 0.0330 - val_acc: 0.9888
Epoch 8/12
60000/60000 [==============================] - 77s 1ms/step - loss: 0.0340 - acc: 0.9896 - val_loss: 0.0251 - val_acc: 0.9919
Epoch 9/12
60000/60000 [==============================] - 78s 1ms/step - loss: 0.0306 - acc: 0.9905 - val_loss: 0.0237 - val_acc: 0.9921
Epoch 10/12
60000/60000 [==============================] - 79s 1ms/step - loss: 0.0288 - acc: 0.9911 - val_loss: 0.0247 - val_acc: 0.9919
Epoch 11/12
60000/60000 [==============================] - 79s 1ms/step - loss: 0.0278 - acc: 0.9911 - val_loss: 0.0250 - val_acc: 0.9918
Epoch 12/12
60000/60000 [==============================] - 78s 1ms/step - loss: 0.0266 - acc: 0.9917 - val_loss: 0.0246 - val_acc: 0.9926
Test loss: 0.024583681731394427
Test accuracy: 0.9926
```

## 마치며...

전체적인 환경설정이 어렵기보단 지치고 힘들었습니다.
현재 지난 설정들을 밀어 virtualenv, pyenv, python설치 등이 초기화되있어서 더 귀찮았습니다.

처음에 python 3.7에서 진행하다가 tensorflow에서 막혀서 당황했습니다. python으로 작업하면 항상 이런 버전문제로 1시간씩은 구글링하는 것 같네요. (예전 Qt5와 python3 사용에서도 이랬던 기억이...)

진행하다가 파이썬 버전을 바꿔서 그런지 주피터 노트북에서 import를 못시켜 system에서 진행했는데, 지금 현재 노트북 상태가 좋지 않은데 무리하게 CPU를 혹사시키는 작업인 것 같습니다. ㅠㅠ

앞으로는 책과 여러 자료를 통해 공부 및 포스팅을 진행해볼까 합니다.
또한 컴퓨터로 그냥 돌리니 너무 오래걸려서 GPU 설정 또는 주피터 노트북을 이용한 설정을 확인해봐야겠습니다.


## Reffernce

맥에서 케라스 설치하기 https://tykimos.github.io/2017/08/07/Keras_Install_on_Mac/
케라스 창시자에게 배우는 딥러닝 https://book.naver.com/bookdb/book_detail.nhn?bid=14069088
