"""Views of Graph.
Inspired by dgl.view
"""

from collections import namedtuple
from dataclasses import replace
from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

NodeSpace = namedtuple("NodeSpace", ["data"])
EdgeSpace = namedtuple("EdgeSpace", ["data", "srcdata", "dstdata"])

class NodeView(object):
    def __init__(self, graph):
        self.graph = graph

    def __getitem__(self, key):
        ntype_idx = self.graph._ntype_invmap[key]
        return NodeSpace(
            data=NodeDataView(
                graph=self.graph,
                ntype_idx=ntype_idx,
            ),
        )

class EdgeView(object):
    def __init__(self, graph):
        self.graph = graph

    def __getitem__(self, key):
        etype_idx = self.graph._etype_invmap[key]
        srctype_idx, dsttype_idx = self.graph.get_meta_edge(etype_idx)
        src, dst = self.graph.gidx.edges[etype_idx]
        return EdgeSpace(
            data=EdgeDataView(
                graph=self.graph,
                etype_idx=etype_idx,
            ),
            srcdata=NodeDataView(
                graph=self.graph,
                ntype_idx=srctype_idx,
                idxs=src,
            ),
            dstdata=NodeDataView(
                graph=self.graph,
                ntype_idx=dsttype_idx,
                idxs=dst,
            ),
        )

class NodeDataView(object):
    def __init__(self, graph, ntype_idx, idxs=None):
        self.graph = graph
        self.ntype_idx = ntype_idx
        self.idxs = idxs

    def __getitem__(self, key):
        res = self.graph.node_frames[self.ntype_idx][key]
        if self.idxs is not None:
            res = res[self.idxs]
        return res

    def set(self, key, data):
        ntype = self.ntypes[self.ntype_idx]
        return self.graph.set_ndata(key=key, data=data, ntype=ntype)

class EdgeDataView(object):
    def __init__(self, graph, etype_idx, idxs=None):
        self.graph = graph
        self.etype_idx = etype_idx
        self.idxs = idxs

    def __getitem__(self, key):
        res = self.graph.edge_frames[self.etype_idx][key]
        if self.idxs is not None:
            res = res[self.idxs]
        return res

    def set(self, key, data):
        etype = self.etypes[self.etype_idx]
        return self.graph.set_edata(key=key, data=data, etype=etype)
