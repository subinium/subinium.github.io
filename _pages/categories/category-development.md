---
title: "Post about Development"
layout: archive
permalink: /categories/development
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.development | sort:"date" %}

{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}
