---
title: "Post about Test"
layout: archive
permalink: /categories/test
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.blog | sort:"date" %}

{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}
