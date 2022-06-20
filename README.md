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

## Design principle
* Pure JAX, end-to-end differentiable and jittable.
* Graphs (including heterographs with multiple node types), metagraphs, and node and edge data are simply pytrees (or more precisely, namedtuples), **and are thus immutable**.
* All transforms (including neural networks inhereted from [flax](https://github.com/google/flax)) takes and returns graphs.
* Grammar highly resembles [DGL](https://www.dgl.ai), except being purely functional.

## Quick start
Implement graph convolution in **five** lines:
```python
>>> import jax.numpy as jnp; import galax
>>> g = galax.graph(([0, 1], [1, 2]))
>>> g = g.ndata.set('h', jnp.ones(3, 16))
>>> g = g.update_all(galax.function.copy_u("h", "m"), galax.function.sum("m", "h"))
>>> W = jnp.random.normal(key=jax.random.PRNGKey(2666), shape=(16, 16))
>>> g = g.apply_nodes(lambda node: {"h": node.data["h"] @ W}) 
```


