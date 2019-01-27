---
title : Project Euler Sidebar 만들기
category :
  - development
tag :
  - blog
  - jekyll
  - liquid tag
  - js
  - vue
sidebar_main : true
use_math : true
header:
  teaser : /assets/images/category/devel.jpg
  overlay_image : /assets/images/category/devel.jpg
---
테마에 sidebar을 추가해봅시다.

## 왜 하는가?

Euler Wiki를 간략하게나 만들고 있는데,
문제로 접근하는 방법이 기존에는 표에서 수를 클릭하는 방법밖에 없었다.

그렇기에 문제를 좀 더 쉽게 접근하기 위해 sidebar은 필수적이었다.
하지만 jekyll 현 테마(또는 jekyll 자체)에서 적용되는 sidebar의 경우에는 목차를 일일히 작성해야한다.

하지만 문제는 600문제인데 하나하나 문제를 업로드할때마다 목차를 작성하면 그건 너무 비효율적이다.

그렇기에 한번 원하는 조건들과 자동화를 이용해 만들어보자!

## 원하는 조건 & 환경파악

우선 코드를 작성하기 위해서 현재 상황과 목표를 정리해보자.

### 원하는 조건

#### 코드
1. 자동화 : 문제를 업로드하면 자동으로 Tag가 만들어져야함.
2. 코드 수정의 용이함 : Format을 만들어 수정할때, 한번에 다 할 수 있게.

#### 그 외
1. 난이도 또는 번호순 정렬
2. sidebar에서 sub-title을 클릭하면 번호와 링크

### 환경 파악

우선 수정해야할 부분을 파악해야한다.
게시물의 경우에는 `_layout`에 속해있는 파일들의 형식으로 따른다.
euler관련 post들은 `_page`에서 PE 디렉토리에 속해있고,
이 모든 게시물은 layout은 `euler_ps`라는 새로운 형식의 layout으로 만들었다.

이 파일의 sidebar 파트의 경우에는 다음과 같은 파일을 include하고 있다.
```html
{ % include sidebar.html % }
```

그리고 sidebar.html에서는 특정 조건에 따라 sidebar을 띄우는 liquid 언어가 있습니다.
그리고 예전에 이 파일을 가져온 분의 기존 코드에 따르면 다음과 같이 만들 수 있습니다.

```html
{ % if page.sidebar_main % }
  { % include nav_list_main % }
{ % endif % }
```

코드를 해석하면 page내에 `sidebar_main`이 true일때, `nav_list_main`을 가져온다.
즉, custom sidebar의 핵심은 page의 요소설정과 새로운 html소스입니다.

``` html
<nav class="nav__list">
  <input id="ac-toc" name="accordion-toc" type="checkbox" />
  <label for="ac-toc">{{ site.data.ui-text[site.locale].menu_label | default: "Toggle Menu" }}</label>
  <ul class="nav__items" id="category_tag_menu">
      <li>
        <span class="nav__sub-title" v-on:click='togglec()'>Categories</span>
        <ul v-show="flag_c">
        { % for category in site.categories % }
           <li><a href="/categories/{ {category[0] | slugify}}" class="">{ {category[0] | capitalize}} ({ {category[1].size}})</a></li>
        { % endfor %}
        </ul>
        <span class="nav__sub-title" v-on:click='togglet()'>tags</span>
        <ul v-show="flag_t">
        { % for tag in site.tags % }
           <li><a href="/tags/#{ {tag[0] | slugify}}" class="">{ {tag[0] | capitalize }} ({ {tag[1].size}})</a></li>
        { % endfor % }
        </ul>
      </li>
  </ul>
</nav>
```

여기서 중요한 것은 `category_tag_menu`와 toggle과 flag 설정이다. (위의 부분은 sidebar의 위치나 글씨체를 설정한 부분이다.) 이 부분을 수정하면 개인 커스텀을 완성할 수 있다.
허나 이 toggle과 flag는 scripts.html에 vue js를 통해 플러그인을 불러오고, 플러그인에 따른 변수 및 함수를 설정해야합니다.

```
_config.yml # 후에 전역변수 설정, 반복문에서 필요
nav_list_PE # 개인 커스텀
scripts.html # 커스텀용 vue 설정
sidebar.html # nav_list_PE 호출 설정
```

## nav_list_PE 만들기

우선 프로그램의 가장 기본적인 틀은 난이도별 분류(5% 간격)와 난이도에 따른 문제(포스팅한 문제들만)로 설정했다.

그렇다면 우선 문제 페이지에는 컴포넌트를 만든다. 컴포넌트는 지킬 기본 설정부분인  header(?)에서 설정하면 된다.

```
---
sidebar_PE : true
diff : 0 # 5, 10 ... 100
num : 000 # 1 ~ 634
---
```

각 컴포넌트는 sidebar을 부르기 위해서, 난이도로 분류하기 위해서 , 문제 번호는 후에 liquid로 가져와 url을 만들기 위해서 만들었다.

```html
{ % for per in site.diff %}
  { % assign cnt = 0 %}
    { % for i in site.pages %}
      { % if i.diff == per %}
        { % assign cnt = cnt | plus : 1 %}
      { % endif %}
    { % endfor %}
  { % assign toggle = "t" %}
  { % assign toggle = toggle | append: per %}
  <span class="nav__sub-title" v-on:click='{ {toggle}}()'> { {per}}% <span style = "font-size:.6em">Difficulty ({ {cnt}}) </span></span>
  { % assign flag = "flag_" %}
  { % assign flag = flag | append: per %}
  <ul v-show="{{flag}}">
    { % for problem in site.pages %}
      { % if problem.diff == per %}
        <li style = "font-size: .7em"><a href="/euler/{ {problem.num}}" class="">{ {problem.title}}</a></li>
      { % endif %}
    { % endfor %}
  </ul>
{ % endfor %}
```

전체적인 알고리즘(?)의 개요는 다음과 같다.

- site.diff는 config.yml에서 설정할 수 있는 환경변수로 미리 5부터 100까지 등록해두었다.

- 굳이 이렇게 한 이유는 아직 포스팅을 하지 않은 난이도가 있더라도 글을 올리기 위해서다.

- liquid내 변수 선언은 assign으로 선언하고, 같은 변수를 부르기 위해서는 assign으로 불러와야한다.

- toggle또는 flag로 만들어 + diff하여 난이도별 toggle과 flag로 string을 만들어서 html 코드에 삽입하였다.

- count를 해줄 수 있는 함수가 없기에 페이지들을 모두 순회하며 난이도가 같은 것의 개수를 셌다.

- 그리고 다시 반복문을 이용해 난이도 별로 서브 li를 만들었다.

## 최종본 nav_list_PE

``` html
<nav class="nav__list">
  <input id="ac-toc" name="accordion-toc" type="checkbox" />
  <label for="ac-toc">{{ site.data.ui-text[site.locale].menu_label | default: "Toggle Menu" }}</label>
  <ul class="nav__items" id="PE_menu">
    <span class="nav__sub-title" style = "font-size = .8em"><a href="/euler/" class="">Back to Check Table</a></span>
      { % for per in site.diff %}
        { % assign cnt = 0 %}
        { % for i in site.pages %}
          { % if i.diff == per %}
            { % assign cnt = cnt | plus : 1 %}
            { % endif %}
        { % endfor %}
        { % assign toggle = "t" %}
        { % assign toggle = toggle | append: per %}

        <span class="nav__sub-title" v-on:click='{ {toggle}}()'> { {per}}% <span style = "font-size:.6em">Difficulty ({ {cnt}}) </span></span>
        { % assign flag = "flag_" %}
        { % assign flag = flag | append: per %}
        <ul v-show="{{flag}}">
          { % for problem in site.pages %}
            { % if problem.diff == per %}
              <li style = "font-size: .7em"><a href="/euler/{{problem.num}}" class="">{ {problem.title}}</a></li>
            { % endif %}
          { % endfor %}
        </ul>
      { % endfor %}
  </ul>
</nav>
```

## 그 외

그 외에도 vue 등등이 있는데 이는 좀 더 공부하고 포스팅해야겠다.

## 마치며

사이트가 점점 내 이상에 가까워지고 있다는 것에 만족.
