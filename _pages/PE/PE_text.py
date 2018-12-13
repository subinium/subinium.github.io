# coding: utf-8
from urllib.request import urlopen
from bs4 import BeautifulSoup

def print_context(x):
	html = urlopen("http://projecteuler.net/problem=%d"%x)
	bsObj = BeautifulSoup(html, "html.parser")
	problem_contents = bsObj.find("div","problem_content")
	return problem_contents.get_text()

def print_code(x):
    html = urlopen("https://raw.githubusercontent.com/subinium/ProjectEuler/master/solved/p%03d.cpp"%x)
    bsObj = BeautifulSoup(html, "html.parser")
    return str(bsObj)


i, diff = input().split(" ")
i = int(i)
diff = int(diff)

file_name = "p%03d.md"%i
f = open(file_name, 'w', encoding='utf8')

f.write( """
---
layout : euler_ps
permalink : /euler/%d
title : Problem %03d
category :
  - algorithm
tag :
  - projecteuler
header :
  overlay_image : /assets/images/pe/%02d.png
use_math : true
num : %d
sidebar_PE : true

---
""" % (i,i,diff,i)
)

f.write(
"""
{% assign num = site.data.PE_problem[page.num] %}

<h1> Problem {{page.num}} : {{num.title}} ({{num.diff}}%) </h1>

<a href = "https://projecteuler.net/problem={{page.num}}">link</a>

## Description

### original
"""
)

f.write(print_context(i))

f.write(
"""
### 간략해석

## Idea & Algorithm

### naive idea

### advanced idea

## source code

"""
)

f.write("``` cpp\n" + print_code(i) + "\n```")
