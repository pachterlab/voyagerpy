---
layout: technology
title: Installing VoyagerPy
name: Install
permalink: /install/
order: 1
toc_header: h2
---

# {{page.title}}

{% assign repo_url=site.github["repository_url"] %}
{% assign repo=site.github["repository_name"] %}

To install VoyagerPy on your system you must clone the [git repository]({{ repo_url }}). Currently, we don't have a proper
release on the `main` branch, so we advise checking out the `dev` branch. In the future, we will make releases to the main branch and have the project available to install via `pip`.

## Clone the repository

```shell
$ git clone {{repo_url}}
$ cd {{repo}}
$ git checkout dev
```

## Create an environment (Optional)

We recommend that you install VoyagerPy in an environment, such as a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```shell
$ conda create -n voyager39 python=3.9
$ conda activate voyager39
```

## Installing VoyagerPy

When you are happy with your environment setup, you can install VoyagerPy using `pip`. Make sure that you are located
in the root directory of the project.

```shell
$ pip install .
```

In most cases, this should work without errors. However, as we have [Geopandas](https://geopandas.org/en/stable/) as a dependency, some users may need a more involved installation of Geopandas and its dependencies.

## Running the example notebooks

If you have a fresh environment and want to run the [example notebooks]({{site.baseurl | append: "/notebooks" }}), there are some packages you might want to install.

```shell
$ pip install jupyter jupyterlab scanpy
```