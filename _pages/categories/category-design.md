---
title: "Post about Design"
layout: archive
permalink: /categories/design
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.design | sort:"date" %}

{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}
