---
title : Xcode에 개인 header 추가하기
category :
  - development
tag :
  - xcode
  - custom
sidebar_main : true
use_math : true
header:
  teaser : /assets/images/category/devel.jpg
  overlay_image : /assets/images/category/devel.jpg
---

xcode에 <bits/stdc++.h>헤더 추가하기

## bits/stdc++.h

이 헤더는 PS를 조금 해 본 사람이라면 아는 헤더로, 아주 강력한 헤더이다.

이 헤더는 PS에 필요한 모든 표준 헤더를 포함하고 있다.

geeksforgeeks.org라는 사이트에서 [설명](https://www.geeksforgeeks.org/bitsstdc-h-c/)은 다음과 같다.


### 단점

- 이 헤더는 GNU C++ 라이브러리의 표준 헤더 파일이 아니다. 따라서 GCC가 아닌 다른 컴파일러로 코드를 컴파일할 경우에는 실패한다.
- 불필요한 헤더를 많이 포함하므로 컴파일 시간이 길어진다.
- 이식성이 떨어진다.

### 장점

- 대회에서 필요한 모든 STL을 선언하는데 시간이 줄어든다.
- 코드가 간결해진다.
- STL을 모두 기억할 필요가 없다.

### 개인적인 코드 환경

왜 안되는지 몰랐는데 다음과 같은 문구를 확인할 수 있었다.

> Mac OS X 10.9+ no longer uses GCC/libstdc++ but uses libc++ and Clang.



## Xcode

xcode에서 c++관련 헤더는 다음과 같은 path에 설정되어 있다.

```
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1
```

다음 path로 가서 bits 디렉토리를 만들고 stdc++.h 파일을 저장하면 된다.


## Terminal

터미널에서 g++로 컴파일하면 다음에서 헤더를 호출한다.

```
/usr/local/include
```

똑같이 다음 path로 가서 bits 디렉토리를 만들고 stdc++.h 파일을 저장하면 된다.

## 추가적 설명

헤더는 다음 링크로 이동하면 다운받을 수 있다. [bits/stdc++.h](https://gist.github.com/eduarc/6022859)

또한 xcode와 terminal에서 `#include <cstdalign>`에서 오류가 떠서 주석처리를 하였다. 왜 안되는지는 아직 모르겠다.
