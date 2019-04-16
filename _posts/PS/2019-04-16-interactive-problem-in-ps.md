---
title : "Interactive 문제란 무엇인가요?"
category :
  - Algorithm
tag :
    - PS
    - subinium
    - Interactive
    - Problem Solving
sidebar_main : true
author_profile : true
use_math : true
header:
  teaser : https://svgsilh.com/svg/799448.svg
  overlay_image : https://svgsilh.com/svg/799448.svg
---

인터랙티브 안 풀어본 사람이 올리는 인터랙티브 글

> 본 글은 codeforces의 **MikeMirzayanov** 님의  [Interactive Problems: Guide for Participants](https://codeforces.com/blog/entry/45307)글을 의역한 글입니다.

저도 인터랙티브의 개념만 알고, 실제로 손 수 풀어본 적은 없습니다.
2018 ACM-ICPC Hanoi Regional에서도 풀이만 제공하고, 코드는 팀원이 짜줬던 기억이 있습니다.

> Thanks to @evenharder, @Lawali

본 포스트는 인터랙티브 문제가 무엇이고, 어떤 식으로 진행되는지 가볍게 소개하는 글입니다.

## What is Interactive Problem?

### Interactive 문제

2018년부터 구글 코드잼의 플랫폼이 바뀌면서 **Interactive** 문제들이 나오고 있습니다.
IOI나 일부 대회들에 간혹 나오는 독특한 형태의 PS 문제입니다.

서로 상호작용을 하며 문제를 해결한다는 점인데, 어떻게 서버와 여러분의 코드가 상호작용을 할까요??

일반적인 Online Judge 사이트에서는 보기 힘든 형태이며, 다음과 같은 사이트에서 일부 문제를 연습할 수 있습니다.

- [oj.uz](https://oj.uz/) - 국내 **최고의 사이트** 중 하나
- [codeforces](https://codeforces.com)
- [codechef](https://www.codechef.com/)
- [Yandex](https://contest.yandex.com/ioi/)
- [DMOJ](https://dmoj.ca/)

### 인터랙티브의 핵심 :

**Interactive** 문제란 입력 데이터가 사전에 결정되지 않고, 여러분의 코드에 따라 입력 데이터가 제공되는 문제입니다. 여기서 사용되는 프로그램을 **Judge** 라고 합니다. Jury라는 표현도 많이 사용하는 것 같습니다.

여러분의 코드와 Judge의 관계는 다음과 같습니다.

<figure class = "align-center"  >
  <div class = "mermaid">
    sequenceDiagram
        Code ->> Judge: First Query
        loop Interactive
            Judge ->> Code: nth State
            Code ->> Judge: nth Query
        end
        Code ->> Judge : Answer(End State)
        Judge ->> Code : AC / WA
    </div>
    <figcaption> Interactive Problem의 가상 흐름도 </figcaption>
  </figure>


여러분의 출력은 Judge의 입력으로 들어가고, Judge는 그에 대한 출력을 여러분에게 다시 제공합니다.
이렇게 상호간의 대화 같은 프로그램은 최종적으로 여러분이 마지막 상태 또는 정답을 출력하면 맞았습니다/틀렸습니다를 알려줍니다.

## 어떤 방식으로 출력을 해야하나요??

### 스트림과 버퍼

우선 인터랙티브 문제를 알기 위해서는 버퍼에 대해서 알아야 합니다.

C++ 프로그램은 콘솔의 입출력을 직접 다루지 않고 **스트림(stream)** 이라는 흐름을 통해 다룹니다. 가상의 중간 매개체 역할이라고 할 수 있습니다.

![string stream buffer](http://www.angelikalanger.com/IOStreams/Excerpt/fig02-09.jpg)

이런 스트림에는 내부에 **버퍼(buffer)** 라는 임시 메모리 공간을 가지고 있습니다. 이런 버퍼 중에서 입출력 버퍼는 입력된 데이터나 출력 장치로 보내기 전에 데이터를 일시 저장합니다. 실제 여러분이 로컬 또는 일반적인 문제에서 PS 진행하는 경우에는 상당히 효율적인 방법입니다.

하지만 이러한 방식은 대화식 채점 프로그램에는 제대로 들어가지 않을 수 있습니다.
입력 버퍼에 의해 judge는 이를 위해 계속 기다리고 있는 상태가 발생할 수 있습니다.
그런 상황을 방지하기 위해 출력을 할 때마다 특수한 조치를 취해줘야 합니다.

### flush

특수한 조치는 `flush`라는 연산입니다. 이를 이용해서 지속적으로 여러분의 출력 버퍼를 비워줘야 합니다. 대부분의 (적어도 PS에 사용되는) 언어는 `flush` 관련 연산을 standard library에 가지고 있습니다.

일부 언어에 대한 예시입니다.

- C : `fflush(stdout)`
- C++ : `cout << flush`
- Java : `System.out.flush()`
- Python : `stdout.flush()`
- Pascal : `flush(output)`

## 인터랙티브 문제의 특징

1. 다른 문제에 비해 입출력이 매우 느립니다. 그렇기에 cin/cout 보다는 scanf/printf를 사용하는 것을 추천합니다. Java에서는 BufferedReader/PrintWriter를 추천합니다.

2. 본인의 코드를 테스트하는 것도 어렵습니다. 여러분이 Jury, interactor의 역할이 되어야 하기 때문입니다.

3. C++ cout에서 endl을 사용하면 자동으로 처리가 됩니다.

## Example : Guess the number

예시는 다음 문제입니다. 문제는 원문을 그대로 가져왔습니다. 문제는 이진탐색을 인터랙티브로 해결하는 문제입니다.

[http://codeforces.com/gym/101021/problem/A](http://codeforces.com/gym/101021/problem/A)

문제를 읽어보신 분들은 위 링크로 들어가 다시 해결해보시는 걸 추천합니다.

### Statement
In this problem there is some hidden number and you have to interactively guess it. The hidden number is always an integer from 1 and to 1000000.

You can make queries to the testing system. Each query is one integer from 1 to 1 000 000. Flush output stream after printing each query. There are two different responses testing program can provide:

- string `<` (without quotes), if the hidden number is less than the integer in your query;
- string `>=` (without quotes), if the hidden number is greater than or equal to the integer in your query.

When your program wants to guess the hidden number, print string `! x`, where x is the answer, and **terminate your program** immediately after flushing the output stream.

Your program is allowed to make no more than 25 queries (not including printing the answer) to the testing system.

### Input
Use standard input to read the responses to the queries.

The input will contain responses to your queries — strings `<` and `>=`. The i-th string is a response to the i-th your query. When your program will guess the number print `! x`, where x is the answer and terminate your program.

The testing system will allow you to read the response on the query only after your program print the query for the system and perform `flush` operation.

### Output
To make the queries your program must use standard output.

Your program must print the queries — integer numbers $x_i$ (1 ≤ $x_i$ ≤ $10^6$), one query per line. After printing each line your program must perform operation `flush`.

Each of the values $x_i$ mean the query to the testing system. The response to the query will be given in the input file after you flush output. In case your program guessed the number x, print string `! x`, where x — is the answer, and terminate your program.

### Solution

문제는 컴퓨터가 생각하고 있는 수를 **질의를 통해서 맞춰라** 입니다.

그렇다면 컴퓨터는 초기에 랜덤 값으로 그 값을 지정합니다. 여기서부터 여러분의 질의는 시작됩니다.
질의를 하고 나서는 버퍼를 항상 비워줘야 합니다.

여러분의 질의에 따라 컴퓨터는 해줄 수 있는 답을 여러분에게 줍니다. 이 문제에서는 여러분이 예측하고 있는 수와 자신이 지정한 수와의 대소 관계를 전달합니다.

전달받은 대소 관계에 관한 입력을 통해 여러분의 코드는 다시 예측을 진행합니다. 그리고 여러분은 최종적으로 컴퓨터가 예측한 숫자를 알았을 때, 마지막 답안에 필요한 최종 질의를 보냅니다. 여기서는 `! x`와 같은 폼으로 답안을 제출하라고 합니다.

그렇게 여러분이 포맷을 맞춰 interactor에 전달하면, 최종적으로 이 문제를 **맞았다** 혹은 **틀렸다** 를 알 수 있습니다. 다음은 정답 예시입니다.

``` cpp
#include <cstdio>
#include <cstring>
using namespace std;

int main() {
    int l = 1, r = 1000000;
    while (l != r) {
        int mid = (l + r + 1) / 2;
        printf("%d\n", mid); // 첫 질의
        fflush(stdout);
        char response[3];
        scanf("%s", response); // interactor의 답

        // interactor에서 온 값을 통해 지속적 질의반복
        if (strcmp(response, "<") == 0) r = mid - 1;
        else l = mid;
    }

    printf("! %d\n", l); // 최종 답안
    fflush(stdout);
}
```

C++ cin과 endl로 좀 더 간단하게는 다음과 같습니다.

``` cpp
#include <iostream>
using namespace std;

int main() {
    int l = 1, r = 1000000;
    while (l != r) {
        int mid = (l + r + 1) / 2;
        cout << mid << endl;
        string s;
        cin >> s;
        if (s == "<") r = mid - 1;
        else l = mid;
    }
    cout << "! " << l << endl;
}
```

## 마무리

보다 구체적으로 인터랙티브 문제를 이해하고 싶으면, `grader`, `checker` 등에 대해서도 아는 것이 좋습니다.
이에 대한 포스팅은 후에 또 해보도록 하겠습니다.

- [dmoj 공식 문서](https://dmoj.readthedocs.io/en/latest/judge/problem_format/)

## Reference

- [codeforces의 **MikeMirzayanov** 님의 Interactive Problems: Guide for Participants](https://codeforces.com/blog/entry/45307)

- [Google Code Jam 설명 문서](https://code.google.com/codejam/resources/faq)
