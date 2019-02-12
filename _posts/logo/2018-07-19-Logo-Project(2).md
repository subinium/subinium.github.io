---
title : Logo Project(2) - Overlay Image & Favicon
category :
  - blog
  - design
tag :
  - logo
  - overlay image
  - favicon
sidebar_main : true
header:
  teaser : /assets/images/category/design.jpg
  overlay_image : /assets/images/category/design.jpg
---

overlay image와 favicon 설정을 완료했다.

## 1. overlay image

### overlay image 출처

적용된 이미지는 subinium각인된 나무볼펜으로, 대학교 1학년 코린이, 알린이, 프린이 시절 인사동에서 샀다.
1학년 가을, 겨울에 날씨가 급건조해지며 나무가 건조해져서 가방 속에서 부러졌으며 현재는 파편조차 남아있지 않다.

이 이미지를 적용한 이유는 overlay이미지는 적당히 흐려도 괜찮고, 흑백느낌이 강하며, 컴퓨터를 강조하며 subinium을 은은하게 티내고 싶었기 때문이다.

### 적용하기

> 이 방법은 minimal mistake theme에서 적용되는 방법입니다.

[공식 solution](https://mmistakes.github.io/minimal-mistakes/layout/uncategorized/layout-header-overlay-image/)

사실 background image인지 알았는데, 생각해보니 백그라운드라 치기엔 부분이미지였다.

대부분의 세팅 방법의 경우에는 [테마 홈페이지 ](https://mmistakes.github.io/minimal-mistakes/)에 다 있지만 명칭을 몰라서 고생했다.

overlay image라는 명칭을 몰라 한참 고생했다. 이 부분은 여러 사이트들의 깃헙을 보면서 파악했다. 홈의 경우, 레포의 루트에 존재하는 `index.html` 또는  `index.md`를 수정하면 되므로 그 파일들만 체크해서 보면된다.

그 결과 두 가지를 알 수 있었는데

1. overlay imaged와 image 차이
  image의 경우에는 넣는 이미지를 사이트 크기에 맞게 적용되며 overlay image의 경우에는 사이트 윗부분에 적당한 크기로 적용된다.

2. excerpt를 이용해 추가적으로 넣고 싶은 문장을 넣을 수 있다.

## 2. favicon 추가

### favicon 이미지 making

favicon은 보통 두가지 방법으로 만드는 것 같다.

1. pixel 찍기
2. 가지고 있는 이미지를 툴을 이용해 축소

원래는 1을 해볼까 싶었지만 픽셀로 찍기에는 툴이 부족하기도 하고, 시간낭비라는 판단하에 포기했다.

2번의 경우에는 이미지를 사이트에 넣고 변환된 결과를 다운받으면 되는데, 이미지는 [코드포스 프로필](http://codeforces.com/profile/GOD_SUBINIUM)을 사용하기로 했다. (*이 이미지에도 스토리가 있는데 우선 생략한다.*)


### 적용하기

favicon은 이 테마에 따로 솔루션이 존재하지 않는다.

이는 테마 홈페이지와 테마 깃헙 [이슈](https://github.com/mmistakes/minimal-mistakes/issues/949)에 설명되어 있다.

하지만 jekyll에서 favicon을 설정하는 방법은 많이 설명 되어 있었다. 나는 이 [블로그](https://medium.com/@LazaroIbanez/how-to-add-a-favicon-to-github-pages-403935604460) 코드를 참조했다.

favicon의 경우에는 head.html에서 수정하면 된다.

head.html은 default.html에 포함관계로 헤드 부분에서 include되고 default.html을 기반으로 모든 레이아웃이 설정되어 있으니 head.html에서 수정하면 된다.

그렇기에 head.html에 다음과 같은 코드를 추가하면 된다.

```shell
<link rel="shortcut icon" type="image/x-icon" href="/favicon.ico">
```

추가하고 바로 적용이 안될 수 있다. 왜인지는 모르겠으나 그런 상황이 발생했다.

우선 3가지의 가능성을 떠올렸다.

1. **favicon.ico가 잘못된 포맷을 가지고 있는 경우**

    파일의 포맷은 이미지 파일을 ico파일로 변경해주는 사이트에서 변경했기때문에 틀릴 가능성이 없다고 판단했다.
2. **코드가 틀리거나 PATH가 잘못된 경우**

    이 경우는 github에서 에러 메세지가 gmail로 오거나 github commit에 빨간색으로 뜨는데 이것도 아니다.
3. **적용이 되었으나, 내가 모르는 이유로 안되는 경우**

    검색해보니 적용에 필요한 요소를 확인해줄 수 있는 [페이지](https://realfavicongenerator.net/favicon_checker#.W09wX9gzbOQ)가 있다. 체크해보니 브라우저에서는 체크가 되어있다고 표현되어 있었다. (그 외로 다양하게 icon이 적용되는 포맷이 많았는데 시간이 되면 추가해야겠다.)
  그래서 크롬을 껏다 켜보니 적용완료!!
