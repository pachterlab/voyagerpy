---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults
layout: default
title: VoyagerPy
---

## {{ "Hello World!" | downcase }}

{{ site.description }}

<div class="container-fluid">
	<div class="collapse navbar-collapse">
		<ul class="navbar-nav me-auto mb-2 mb-lg-0">
		{% for nav in site.navitems %}
			<li class="nav-item">
			<a href="{{ nav.url }}"> {{ nav.name }} </a>
			</li>
		{% endfor %}
		</ul>
	</div>
</div>