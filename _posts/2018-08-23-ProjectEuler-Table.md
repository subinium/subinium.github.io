---
title : Project Euler Checklist 만들기
category :
  - blog
  - development
tag :
  - html
  - css
  - projecteuler
sidebar_main : true
use_math : true
toc : true
header:
  teaser : /assets/images/category/devel.jpg
  overlay_image : /assets/images/category/devel.jpg
---

본격적으로 스타트하기 전에 Project Euler세팅을 해봅시다.

## 프로젝트 오일러를 하기전에!

### INTRO
지속적인 작업을 위해서는 초기 설정이 매우 중요하다.
물론 초기 설정은 매우 귀찮다. 그러나 공부 안되는 날에는 책상 정리하듯이 초기설정을 해보았다.

개인적으로 사이트나 블로그 등 사람들에게 보여지기 위한 플랫폼에서 가장 중요한 것은 UI라고 생각한다.
그렇기 때문에 많은 PS사이트(알고스팟, Codeforces, Topcoder) 중에서 BOJ를 좋아한다.
(내가 능력만 된다면 살짝 바꾸고 싶은 부분들이 많지만 **충분히 이쁘다**. 블로그 하나도 이렇게 만들기 어려운데..)

이번에는 틀린 문제, 최적화가 필요한 문제, 포스팅한 문제 등 문제들을 분류하는 체크표를 만들 예정이다.
문제에 대한 기록과 체크리스트를 만들겠다는 생각은 몇몇 사례들에서 비롯되었는데, 다른 사람들도 알면 좋을 것 같은 사이트 및 자료다.

1. **@koosaga** 님이 만든 [Checklist for OI problems](http://codeforces.com/blog/entry/59422)
2. **@koosaga** 님이 알려준 [Atcoder Checklist](https://kenkoooo.com/atcoder/?user=subinium&rivals=&kind=category)
3. **@kajebiii** 님 블로그 [BOJ 궤짝](https://ps.kajebiii.com/chest/boj/)
4. **@godmoon00** 님이 만든 [BOJ 전적](https://acmicpc.cc/)
5. **@cubelover** 님이 만들고 **@tncks0121** 님이 운영하는 [BOJ Tier](https://koosaga.oj.uz/)

최근에 느낀건데 나는 문제를 푸는 것과 그 사실을 기록하는 것을 매우 좋아한다. 그렇기 때문에 문제리스트와 그 푼 문제를 한눈에 볼 수 있는 PE에 더 혹했던 것 같다.

그리고 풀이를 찾기위해 검색하던 중 아주 마음에 드는 [블로그](https://euler.stephan-brumme.com/)를 만났다.
이 블로그에서 가장 빛나는 것은 *Heatmap* 이라는 생각이 든다.

![heatmap](/assets/images/pe/heatmap.png)

처음에는 PE에서 제공하는 플러그인 또는 Hackerrank에서 제공하는 플러그인인줄 알았는데, 아무도 없는 걸 봐서는 직접 만든 것이다.

마음같아서는 그대로 복붙하고 싶었지만 css를 한번에 복붙하기도 힘들고, 내가 원하는 분류는 따로 있고, 디자인과 색감이 영 마음에 안들었기때문에 이 것을 만들어볼까 한다.

### 간략 계획도

그렇다면 이 표를 어떻게 만들 거고, 어디에 배치하고, 어떤 식으로 설정할지 정했다.

1. 표 만들기 (html)
2. 디자인 하기 (css)
3. includes에서 liquid tag로 설정 (includes파일 설정)
4. 메뉴에 Euler-Wiki 만들기 (data파일 navigation, pages에서 메뉴 생성)
5. posting하면 wiki에 쌓이는 방식.

5번은 앞으로 차근차근해야하고 전체적인 1-4까지를 진행해보았다.

## 표만들기 & 디자인하기

### 표 구상하기

표를 만들기 위해서는 문제에 대한 분류가 우선이다. 그래야 html작성시에 class분류가 편하기 때문이다.

문제에 대한 분류는 6 단계로 분류해보았다.

1. 기본값
2. 시도한 문제
3. 맞은 문제. 최적화 필요 또는 풀이 재작성 필요
4. 맞은 문제. 풀이 존재 및 최적화 완료
5. 포스팅 완료. 그러나 최적화 필요한 문제
6. 포스팅 완료. 최적화도 만족

표는 20개씩 5줄로 100문제씩 한 table에 만들면 된다.
그럼 약 6.5그룹 정도 나온다.

### 파이썬으로 표 작성하기

표는 파이썬 파일입출력으로 쉽게 만들었다.

가장 기본적인 아이디어는 상위단계에 체크한다면, 하위단계에서 체크된 문제는 생략된다는 점이다.

2-6까지 리스트는 풀거나, 시도할 때마다 각각에 해당하는 txt파일에 문제 번호를 적어준다. 그리고 list에 각각 해당하는 단계를 적어주고, 단계 별로 class를 설정해주면 된다.

표를 만들때는 table, tr, td 정도에 class를 설정해주었는데, 그냥 만드니 조금 아쉬워서 a tag를 이용해 문제와 연결시켜두었다. 후에 포스팅하면 포스팅한 문제로 연결할 수 있게 코드를 수정할 에정이다.

숭고한에서 문제 데이터 만들면서 파일입출력이 조금 익숙해져서 코드는 금방짰다.

#### 소스코드

다음은 파이썬 코드다. 소스코드에 설명부 입출력 파트는 생략했다.

{% highlight python %}
SZ = 634

problem_list = [0] * (SZ+1)

# tried : i read problem but i don't know
tried = open("tried.txt","r")
for i in tried.read().split():
    problem_list[int(i)] = 1

# optimazation : accepted but it needs optimazation (1 min solution)
opt_need = open("opt-need.txt","r")
for i in opt_need.read().split():
    problem_list[int(i)] = 2

# accepted : accepted
accepted = open("accepted.txt","r")
for i in accepted.read().split():
    problem_list[int(i)] = 3

# posted : blog posting ok
# posted but still need optimization or need more proof
posted_unclear = open("posted-unclear.txt","r")
for i in posted_unclear.read().split():
    problem_list[int(i)] = 4

# posted and enough optimization
posted_clear = open("posted-clear.txt","r")
for i in posted_clear.read().split():
    problem_list[int(i)] = 5

table = open("PE_table.html","w")

class_type = ["default", "tried","opt-need","accepted","posted-unclear","posted-clear"]

for i in range(1, SZ+1):
    if i % 100 == 1:
        table.write("<table class=\"euler\">\n\t<tbody>")
    if i % 20 == 1:
        table.write("\t\t<tr>\n")
    flag = problem_list[i]
    # https://projecteuler.net/problem=66
    table.write("\t\t\t<td class="+"\""+class_type[flag]+"\""+"><a href = \"https://projecteuler.net/problem="+str(i)+ "\">"+str(i)+"</a></td>\n")
    if i % 20 == 0:
        table.write("\t\t</tr>\n")
    if i%100 == 0:
        table.write("\t</tbody>\n</table>\n")

table.write("\t\t</tr>\n\t</tbody>\n</table>\n")

{% endhighlight %}


### CSS 다루기

css는 그냥 취향에 맞춰서 하면 된다. rgb값은 구글에서 color palette 를 검색한 후, 어떤 [사이트](https://www.duolingo.com/)의 색상 표를 참조했다. 마음에 드는 색이 많아서 설정했다.

다음 table관련 css는 `_sass/minimal-mistakes/_tables.scss`에 있다.

#### 배운점 / 응용점
- 기존의 버튼의 디자인에서 모서리 처리 추가했다. (border-radius)
- a tag에서 색상 설정에서 inherit을 공부했다.
- rgb값만 한 50가지 테스팅해본듯.
- css class설정 방법을 까먹었는데, 이제 안까먹을거같다.

#### 2차 수정

- mobile에서 깨지는 원인은 width : 100%에서 칸을 맞추기위해 계속 깨짐
- 그걸 없애고 폰트사이즈와 패딩을 모두 %단위로 설정하여 모바일에서 깨지지 않게 설정
- 글씨체 cursive가 마음에 들어서 변경

#### 소스코드
```css
.euler {
   margin : 0.3em;
   //border : 1px solid black;
   cursor : pointer;
   border-collapse: separate;
   border-spacing: 1px;
   td {
     border-radius: $border-radius;
     font-family: cursive;
     font-size: 70%;
     padding: 0.3% 0% 0.3% 0%;
     border: 0.1px solid grey;
     text-align: center;
     width: 40px;
     table-layout: fixed;
     word-break : break-all;
   }
   .posted-clear {
     background: #8EFF00;
     color: #000000;
     font-weight: bold;
   }
   .posted-unclear {
     background: #8EFF00;
     color: #FA811B;
   }
   .opt-need {
     background: #FFCF7A;
     color: #FA811B;
   }
   .accepted {
     background: #BCE9FF;
     color: #1CB0F6;
   }
   .tried {
     background-color: #FF9797;
     color: #D33131;
   }
   .default {
     background-color: #F0F0F0;
     color: #777777;
   }

   th {
     padding: 0;
     text-align: center;
   }
   a{
     color: inherit;
     text-decoration: none;
     table-layout: inherit;
   }
}
```

## Liquid Tag 설정

이건 MathJax하면서 익혔기때문에 쉬웠다. 다만 이번엔 파이썬 코드로 계속 html파일을 수정해야기때문에 디렉토리를 만들고 파이썬 파일과 체크 txt를 만들었다.

현재는 다음과 같은 liquid tag로 table에 부를 수 있다.

```
{ % include pe_support/PE_table.html % }
```

## Euler Wiki 메뉴 만들기

메뉴는 `_data/navigation.yml`에 추가하면 되고, 그에 맞는 페이지는 `_pages`에 html또는 md파일을 만들어 permalink만 초기에 설정했던대로 추가하면 된다.


## 결과물!!

다음은 결과값이다. 아직 맞은 문제만 체크해서 부실하다. 최적화 못한 문제도 빨리 체크해야하는데 이번 주 중에 몇 개 해야지.
설명문과 Euler Image는 위에 파이썬 코드에 추가해서 함께 작성되게 하였다.

![screenshot](/assets/images/PE/heatmap_sb.png)

## 마치며

이제 다 만들었으니, 포스팅을 해야한다!

블로그에 풀이는 빠른 검색이 되게 만들고 싶으므로 다른 사이트도 풀이를 올릴때는 메뉴를 만들어야겠다. 사이드바도 만들어야하고 할게 참 많다!!

어떤식으로 구성해야 블로그가 이뻐질까...
