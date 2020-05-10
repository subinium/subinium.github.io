---
title : "Linux에서 Process관련 Command 명령어"
category :
  - development
tag :
  - cmd
  - top 
  - htop
  - kill
  - ps
sidebar_main : true
use_math : true
header:
  teaser : /assets/images/category/devel.jpg
  overlay_image : /assets/images/category/devel.jpg
---
리눅스에서 MultiProcessing하다가 정리하고 싶어서 쓴 글

리눅스는 멀티프로세싱이 가능한 OS입니다. 

종종 리눅스 환경에서는 프로세스를 확인할 일이 많습니다.

- 컴퓨터의 CPU와 메모리를 과하게 잡아먹는 프로세스 확인 및 종료
- 멀티 프로세스가 제대로 이루어지는지 확인 (특히 GPU)
- 제대로 프로세스가 진행되고 있는가

이런 상황에서 몇 가지 명령어를 알아두면 좋습니다.

## PS

`ps` 명령어는 실행되고 있는 프로세스의 리스트를 보여줍니다.
보통 `ps aux`와 `ps -ef | grep [something]` 많이 사용하고는 하는데, `aux` 의미는 다음과 같습니다.

- `a` : 모든 유저의 프로세스
- `u` : 유저 정보 출력
- `x` : 터미널 외의 모든 프로세스

그리고 `-ef`의 의미는 다음과 같습니다.

- `-e` : 커널 프로세스를 제외한 모든 프로세스
- `-f` : full listing, 모든 정보를 보여줍니다.

> 그리고 `grep`은 plain-text를 찾아주는 명령어입니다. `pggrep`으로 유사하게 사용할 수 있습니다.

결론적으로는 모든 프로세스를 확인하고 그 중에서 원하는 프로세스를 찾아 끄거나 확인하거나 하는겁니다.

## top

`top`은 ps와 유사하며 지속적으로 프로세스의 현황을 살펴볼 수 있습니다. 
자판에서 `q`를 눌러 탈출할 수 있습니다.

## kill

`kill`은 프로세스를 끄는/죽이는 명령어입니다.

`kill pid`로 pid, 즉 프로세스의 숫자 아이디로 끌 수 있습니다.

하지만 `kill -9 pid`라는 표현으로 많이 사용하는데 그 이유는 `kill` 명령어는 여러가지 방식으로 프로세스를 끄는데, 그중 9번이 즉시 종료하는 명령이라 그렇습니다.