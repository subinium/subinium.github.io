---
title: "Post about Algorithm"
layout: archive
permalink: /categories/algorithm
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.algorithm | sort:"date" %}

{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}
