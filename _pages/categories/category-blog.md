---
title: "Post about Blog"
layout: archive
permalink: /categories/blog
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.blog | sort:"date" %}

{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}
