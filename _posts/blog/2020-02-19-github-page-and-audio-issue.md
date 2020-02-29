---
title : "Github Page로 Audio Demo 만들기와 Issue"
category :
  - development
tag :
  - audio tag
  - github page
sidebar_main : true
use_math : true
header:
  teaser : /assets/images/category/devel.jpg
  overlay_image : /assets/images/category/devel.jpg
---

다양한 Github Page를 만드는 연습중

- [Intern 보고서](https://subinium.github.io/Voice-Conversion)를 쓰는 과정에서 Github페이지에 Audio를 올리며 여러가지 이슈가 있어서 기록용으로 글을 남깁니다.
- 추가로 단순한 github page를 사용한 demo에 사용할 수 있는 Tip을 일부 정리합니다.
- Audio를 업로드하며 issue가 있었으며 그에 대한 글도 함께 씁니다.

## Tips for Github Page

최근 Github Page Custom과 Web 개발을 다시 시작하기 위해 간단한 작업으로 CSS 처리를 모두 밑바닥부터 시작하는 것을 목표로 잡았습니다.
그 과정에서 사용하였고 저장해둘만한 내용을 소개합니다.

### Highlight

> 출처 : [Jekyll 기반의 GitHub Page 생성(4)](https://moon9342.github.io/jekyll-rouge)

Jekyll을 설치하거나, 기존 Theme을 불러와서 Markdown을 사용할 수 있겠지만 기본적으로 설치하는 방법은 다음과 같습니다.

``` shell
gem install rouge
```

rouge는 pure-ruby syntax highlighter입니다.
그리고 `config.yml` 파일은 다음과 같은 문구를 추가합니다.
보통 `_config.yml`은 site 내부의 전역변수를 선언하는 역할이라고 생각하면 편합니다.

``` yml
# _config.yml파일

markdown : kramdown
highlight : rouge
```

그리고 원하는 theme을 따라 css 파일을 만듭니다.
개인적으로는 monokai와 dracular를 가장 좋아합니다.

``` shell
rougify style monokai.sublime > ./css/syntax.css
```

파일 path는 원하는 곳에 설정하고 나중에 layout에서 불러올 때 신경쓰면 됩니다.

``` html
<link rel="stylesheet" href="{{ "/css/syntax.css" | prepend: site.baseurl }}">
```

단순하게 이 방법만 쓰면 너무 highlight 부분이 타이트하니, padding을 추가하고 취향에 따라 border-radius를 추가하면 좋습니다.

### Fonts

폰트는 font-awesome을 오프라인으로 다운받아 사용할 수 있으나 구글웹폰트를 그대로 사용하는 것도 좋습니다.
이전 강의 자료에서 **Noto Sans KR** 글씨체를 많이 사용하여, 이번에도 Noto Sans KR 글씨체를 사용하였습니다.

``` html
<link
    href="https://fonts.googleapis.com/css?family=Noto+Sans+KR&display=swap&subset=korean"
    rel="stylesheet"
  />
```

이를 html header 부분에 넣어주고, css에서는 다음과 같이 모든 요소에 적용하였습니다.

``` css
* {
  font-family: "Noto Sans KR", "Times New Roman", Times, serif;
}
```

### Local Test

local test를 shell code로 저장하고 사용하는 방법을 터득했습니다. PRML 코드를 살펴보다 실행코드가 있어 가져왔습니다.
포트를 따로 설정하여 실행할 수 있는지 몰랐는데 유용한 기능입니다.

디렉토리 루트에 `run_local_jekyll.sh`를 만들어두고 터미널에서 `source run_local_jekyll.sh`을 실행해주면 됩니다.
(window 환경은 잘 모르겠네요)


``` shell
#!/bin/bash

jekyll serve -w -P 10000 --incremental
```

또한 음원은 `./assets` 디렉토리에 옮겨두고 이를 호출하는 방법으로 진행하였습니다.

## Audio

### Audio 파일을 올리는 방법

Github Page에서 오디오 파일을 올리는 방법은 다른 html에서 Audio를 올리는 것과 동일합니다.
다음과 같은 html 코드로 작성 가능합니다.

``` html
<audio controls>
    <source src='./assets/test.wav'>
</audio>
```

- `source`는 연속으로 N개를 나열할 수 있습니다.
- `controls` 속성은 오디오 재생, 볼륨, 일시 정지 컨트롤을 브라우저에서 제공합니다.

audio type을 따로 지정해줄 수 있으나 .flac파일과 .wav파일에서는 상관없이 불러오기가 가능했습니다.

### 어떤 이슈가 발생하였나?

**1. Audio Loading Error**

Audio의 파일이 많아지니 특정 wav 파일이 **loading이 안됩니다.**

local test와 push한 후 github page에서도 확인해보았으며, 둘 모두 음원을 불러오는데 실패하였습니다.

local에서는 음원의 불러진 듯 보이지만, 재생을 하는 경우 시작 0.5~1초 정도의 소리만 들리는 현상이 발생하였습니다.

github page에서 확인한 결과, 음원이 불러지지 않는 것을 확인했습니다.


**2. Audio CSS**

audio 기본 color 값, border-radius 등을 수정하고자 했으나 수정이 전혀 되지 않았습니다.

### 해결방법

**1. rawgithub을 통한 호출과 preload 속성 사용**

우선 첫번째로 시도한 방법은 rawgithub url을 사용하여 호출한 방법입니다.

기존 audio를 불러오는 url의 경우는 다음과 같은 포맷을 가지고 있습니다.

``` shell
./assets/test.wav
```  

불러오는 방식의 변화를 시도해보았습니다. 기존에 방식이 root 디렉토리에서 audio 파일을 불러왔다면, 다음 방식은 github repo에서 link로 불러오는 방식을 사용했습니다.

``` shell
https://raw.githubusercontent.com/subinium/Voice-Conversion/master/assets/test.wav
```

이 방식으로 진행했을 때, local에서는 파일이 불러지는 것을 확인했습니다. 하지만 page에서 불러와지지 않았습니다.

두 번째로 시도한 방법은 preload 속성을 사용하여 해결했습니다. page에서 각 audio file에 대한 정보를 console 창으로 살펴본 결과 404 Error가 나왔습니다.
그리고 load되지 않은 audio 들은 load된 audio 들보다 용량이 큰 audio 였습니다.

보통 이런 경우는 API 비동기 처리와 관련되어 있다고 보면 됩니다. Load가 되기 전에 session을 종료하여 Error가 발생하는 경우입니다.
그래서 audio 관련 속성을 찾아본 결과 `preload` 라는 호출을 기다리는 속성을 찾을 수 있었고, `preload='autio'` 를 audio tag에 추가하니 Load가 느리지만 모두 성공적으로 이뤄졌습니다.


**2. 불가능**

html에서 사용하는 audio tag의 값은 width, height, margin, padding을 제외하고는 css로 따로 수정할 수 있는 부분이 없었습니다.

보통 javascript를 사용하여 custom playbar를 만드는 경우가 대다수인 것 같습니다. 시간이 조금 더 있었다면 이도 custom으로 만들었겠지만, 단순한 중간보고 였기 때문에 시도하지 않았습니다.