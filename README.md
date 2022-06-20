Graph Learning with JAX
========================
[//]: # (Badges)
[![CI](https://github.com/yuanqing-wang/galax/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/yuanqing-wang/galax/actions/workflows/CI.yml)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/yuanqing-wang/galax.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/yuanqing-wang/galax/context:python)
![pypi](https://img.shields.io/pypi/v/g3x.svg)
[![docs stable](https://img.shields.io/badge/docs-stable-5077AB.svg?logo=read%20the%20docs)](https://galax.wangyq.net/)

Galax is a graph-centric, high-performance library for graph modeling with JAX.

## Installation
```
> pip install g3x
```

## Quick start
Graph convolution in a few lines:
```
>>> import jax.numpy as jnp
>>> import galax
>>> g = galax.graph(([0, 1], [1, 2]))
>>> g = g.ndata.set('h', jnp.ones(3))
>>> g = g.update_all(galax.function.copy_u("h", "m"), galax.function.sum("m", "h"))
```


