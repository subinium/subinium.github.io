---
title : Project Euler Update(Tooltip, format)
category :
  - development
tag :
  - data crawling
  - html
  - css
  - blog
  - jekyll
  - yml
sidebar_main : true
use_math : true
header:
  teaser : /assets/images/category/devel.jpg
  overlay_image : /assets/images/category/devel.jpg
---

데이터 크롤링 + html/css + liquid 의 결과물.

## 업데이트 내용

### 내용

우선 다음과 같은 업데이트를 하였는데, 업데이트 내용은 다음과 같다.

1. tooltip 추가

2. tooltip 내용으로 problem title과 difficulty 제공

3. Project Euler 포스팅 format file 설정 변경

### 왜 했는가?

1. 문제를 들어갈때, 어떤 문제인가 정보가 너무 적어 정보 제공의 목적으로 데이터를 추가해보았다.

2. 정보의 경우에는 제목과 난이도 말고는 너무 데이터가 변동적이라 받기 어려웠다.

3. 포스팅할때 수정해야할 변수가 너무 많아 한번에 설정하였다.

### 필요한 작업 목록

1. tooltip 생성 방법 (html/css + js)

2. Project Euler 문제 제목 및 난이도 목록

3. 1, 2와 그 외 데이터들을 좀 더 쉽게 연계하기 위한 방법

## Tooltip 생성방법

[[CSS] 툴팁 기능을 CSS만 사용해서 구현하는 방법과 원리](http://programmingsummaries.tistory.com/369) 을 참조하였다.

이는 JS로 구현할 수도 있지만 JS보다 간단한 방법이 있어서 긁어왔다. 후에 조금 더 손볼 예정이지만 우선은 약간의 수정만 거치고 바로 적용하였다.

이번 기회를 통해 CSS Selector를 설정하는 방법을 알았다.

CSS구현의 가장 포인트는 hover과 after을 통한 가상 선택자와 가상 자녀 요소이다.


이런 custom css의 경우에는 `_sass/minimal-mistakes/_custom.scss`에 추가하면 된다.

추가적으로 설정을 하였지만, 이 부분은 생략한다.

## 데이터 연계 방법

jekyll 설명을 보니 다음과 같은 기능을 알 수 있었다.

site 변수를 config.yml에서 설정할 수도 있지만, site.data에 yml 파일을 만들어 변수를 호출할 수 있었다.

그렇기에 각 포스트를 num 변수 하나만 있으면 문제번호, 제목, 난이도 등은 알아서 설정되게 해두는 방법을 선택하였다.

그렇기에 yml파일을 만드는 것만 남았다.

## 문제 제목 및 난이도 목록

문제 제목 및 난이도 목록을 가져오기 위해서는 3가지 방법을 생각하였다.

1. 푼 문제에 대해서만 하나씩 추가한다.

2. https://projecteuler.net/show=all 에 들어가 모든 문제 데이터를 html로 받아 파싱한다.

3. 문제 주소가 모두 https://projecteuler.net/problem= 의 형식인걸 이용해 크롤링한다.

문제에 대한 갱신이 있을 수도 있기에 코드를 계속 반복해서 쓸 수 있게 3번을 작성하였다.

### projecteuler.net 크롤링하기

이미 학교 수업 중 DB Term Project과목을 하며 Beutiful Soup를 사용해본적이 있기때문에 이를 이용했다.

Beutiful Soup은 파이썬에서 웹크롤링을 할 때 사용하는 라이브러리다. 사용법도 매우 간단하므로 쉽게할 수 있다. 아래는 코드와 같다.

또한 문제명은 h2 tag에 문제에 대한 information은 h3 tag에 존재하였고, 이를 이용해 데이터를 가져오고 파싱하여 yml파일로 바로 출력하였다.

``` python
# coding: utf-8
from urllib.request import urlopen
from bs4 import BeautifulSoup

def web_crawling(n):
	f = open('PE_problem.yml', 'w', encoding='utf8')
	for x in range(1, n):
		html = urlopen("http://projecteuler.net/problem="+str(x))
		bsObj = BeautifulSoup(html, "html.parser")
		problem_title = bsObj.find("h2")
		f.write(str(x)+":\n")
		f.write(" title : "+problem_title.get_text()+"\n")
		problem_info = bsObj.find("h3").get_text().split()
		f.write(" diff : "+problem_info[-1][0:-1]+"\n")
	f.close()

SZ = 635

web_crawling(SZ)
```

recent문제의 경우에는 난이도가 아직 설정되지 않았기에 recent로 따로 수정했다.

아쉬운점이 있다면 projecteuler 사이트 자체에 response 시간이 불규칙하게 오래걸릴때 python에서 timeout error이 떴다. 해결책으로 3번 정도 나누어서 파일을 만들고 합쳤다.

주의점이 있다면 yml파일은 tab을 인식하지 못하니 '\t'가 아닌 ' '로 indent를 맞춰야한다.

## 최종 마무리

### format.md

그렇기에 풀이같은 경우에는 다음과 같은 포맷으로 좀 더 편리하게 바꿨다.

```html
{ % assign num = site.data.PE_problem[page.num] %}

<h1> Problem { {page.num}} : { {num.title}} ({ {num.diff}}%) </h1>

<a href = "https://projecteuler.net/problem={ {page.num}}">link</a>
```

또한 tooltip의 경우에도 마찬가지로 html 생성 코드를 다음과 같이 수정하였다.

``` python
table.write("\t\t\t<td class="+"\""+class_type[flag]+"\" data-tooltip-text = \"{ {site.data.PE_problem["+ str(i) + "].title}} ({ {site.data.PE_problem["+ str(i) + "].diff}}%)\" > "+ "<a href =" + link +str(i)+ "\">"+str(i)+"</a></td>\n")
```

## 마치며

이제 뭐를 업데이트할 수 있을까 고민이다. 조금 더 편하게 만들 수 있으면 좋겠다.
