---
title : Project Euler 이제 시작합니다!
category :
  - algorithm
tag :
  - math
  - projecteuler
  - algorithm
sidebar_main : true
use_math : true
header:
  teaser : /assets/images/category/algo.jpg
  overlay_image : /assets/images/category/algo.jpg
---

프로젝트 오일러 풀이 이제 시작합니다!

## Project Euler란?

> 최근 페이스북에 포스팅하긴 했지만 이제 포스팅할 예정이기 때문에 정리해서 쓴다.

[원래사이트](projecteuler.net) / [한국사이트](euler.synap.co.kr)

[페북포스팅 - 수비니움의 코딩일지](https://www.facebook.com/subiniumscd/posts/219610695386334)

페북포스팅보다 조금 더 자세하게 사이트를 소개하자면

1. 수학문제 사이트다.
  - 필요한 연산이 많아 컴퓨터 능력이 필요하다.
  - 정수론, 확률, 기하, 조합, 알고리즘 등 다양한 능력을 요구한다.

2. 정답만 출력하면 된다.
  - 시간제한이 없다. 최근에 3분넘게 Naive하게 푼 문제가 있다.
  - 언어또한 제한이 없다. 본인은 BigInteger가 무서울때 Python, 보통은 C++을 쓴다.

3. 푼 사람에 따라서 문제의 난이도가 정해진다. 난이도에 따라 문제 선택을 하면된다.
  - 5%와 10% 문제는 난이도가 납득되나 15% ~ 40%에는 난이도가 조금 섞여있다.
  - 문제는 난이도 순이 아니므로 난이도 순으로 정렬해서 푸는게 좀 더 좋다.

4. 사이트에는 Level과 Award가 있다.
  - Level은 25문제에 1씩 오른다. 4Level중간인 112문제는 상위 1%다.
  - Award는 특정한 조건을 만족해야한다. 1%에 들기, 앞에서 100문제 풀기 등등이 있다.
  - 큰 의미는 없는데 동기부여가 된다.

한국랭킹 1페이지에 들어가면 포스팅을 시작하려 했는데, 드디어 목표를 이뤘기에 이제 문제를 풀며 포스팅을 진행하려고 한다.
그리고 한국 1페이지에 들어가면 전체 랭킹 1%에 든다고 한다. (근데 그게 6400명정도 된다. 랭작하기 어렵군.)

![1page](/assets/images/pe/1page.png)
![one percenter](/assets/images/pe/one-percenter.png)

## 어떤 공부를 해야할까..?

수학문제가 대다수인데 보통 쓰이는 몇개가 있다. 기본적인 아이디어들은 다음과 같다.

### 전처리하기(에라토스테네스의 체)

숭고한때 강의한 내용이기도 한데, 생각보다 에라토스테네스의 체는 전처리하기에 좋은 방법이다.
많은 문제는 수 범위가 크기 때문에 전처리를 필요로 한다. 특히 소수, 약수 등의 전처리는 프로젝트 오일러에서는 필수다.

개인적으로는 에라토스테네스의 체를 이용한 전처리를 추천한다.
에라토스테네스의 체는 시간 복잡도도 약 O($nloglogn$)이기 때문에 범위가 1천만 이하라면 전처리하기 좋다.
그렇다면 어떤 전처리를 할 수 있을까?

#### 1. 소수 리스트, 소수 체크

{% highlight cpp %}
const int SZ = 1e7;

int arr[SZ];
vector<int> prime;

void era(){
  arr[0] = arr[1] = 1;
  for(int i = 2 ; i < SZ ; i++){
    if(arr[i]) continue;
    prime.push_back(i);
    for(int j = 2*i ; j < SZ ; j+=i){
      arr[j] = 1;
    }
  }
}
{% endhighlight %}

#### 2. 약수의 개수, 약수의 합

{% highlight cpp %}

int div_cnt[SZ];
int div_sum[SZ];

void era_div(){
  for(int i = 1 ; i < SZ ; i++){
    for(int j = i ; j < SZ ; j+=i){
      div_cnt[j]++;
      div_sum[j]+=i;
    }
  }
}
{% endhighlight %}

#### 3. 서로소 개수

서로소의 개수는 의외로 쓰는 문제가 있으니 이런 식으로 전처리하면 좋다.

{% highlight cpp %}

int phi[SZ];

void era_phi(){
  for(int i = 1 ; i < SZ ; i++) phi[i] = i;
  for(int i = 1 ; i < SZ ; i++){
    if(phi[i]!=i) continue;
    for(int j = i ; j < SZ ; j+=i){
      phi[i] = phi[i]/i*(i-1);
    }
  }
}
{% endhighlight %}

### GCD, POW함수

최소공배수와 지수승은 수학에서 가장 많이 쓰는 함수이다. 자신만의 스타일로 코딩하자.

{% highlight cpp %}
typedef long long ll;

// 유클리드 호제법
ll gcd(ll a, ll b){
  return a%b?gcd(b,a%b):b;
}

// a^b % mod, log복잡도로 지수승 구하기
ll pow(ll a, ll b, ll mod){
  ll ret = 1;
  while(b){
    if(b&1) ret = ret*a%mod;
    a = a*a%mod;
    b >>= 1;
  }
  return ret;
}

{% endhighlight %}

소수에 대한 모듈러 인버스를 구할때 power함수를 사용하므로 항상 익혀두자.

### DP, Greedy

우선 10%난이도에서는 DP는 나이브한 복잡도로 나오는 것 같으니 기본적인 DP연습이 좋다.
Greedy문제는 딱히 특별한 풀이가 있는 것은 아니지만, 우선순위 큐는 알면 좋을 듯하다.

### 상수설정

다른 문제들과 다르게 TC가 많이 존재하진 않지만 문제 대부분에는 예시를 주기 때문에 예시값을 테스트하기 위해서 계속 수를 적어주는 것보다는 상수를 설정해서 상수만 변경하는게 좋은 방법이다. 개인적으로는 const를 이용해 상수를 설정한다.

## 마치며

프로젝트 오일러 한국어 포스팅이 얼마 없으니 프로젝트 오일러 위키(?)가 되고 싶다.

수학말고 알고리즘도 공부해야하는데 참 시간은 없고, 하고 싶은 건 많다~
