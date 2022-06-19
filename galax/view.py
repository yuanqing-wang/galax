"""Views of Graph.

Inspired by dgl.view
"""

from collections import namedtuple
import jax.numpy as jnp

NodeSpace = namedtuple("NodeSpace", ["data"])
EdgeSpace = namedtuple("EdgeSpace", ["data", "src", "dst"])


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
            src=NodeDataView(
                graph=self.graph,
                ntype_idx=srctype_idx,
                idxs=src,
            ),
            dst=NodeDataView(
                graph=self.graph,
                ntype_idx=dsttype_idx,
                idxs=dst,
            ),
        )

    def __call__(self, key=None):
        etype_idx = self.graph.get_etype_id(key)
        src, dst = self.graph.gidx.edges[etype_idx]
        return src, dst

class NodeDataView(object):
    def __init__(self, graph, ntype_idx, idxs=None):
        self.graph = graph
        self.ntype_idx = ntype_idx
        self.idxs = idxs

    def __getitem__(self, key):
        res = self.graph.node_frames[self.ntype_idx][key]
        if self.idxs is not None:
            res = jnp.take(res, self.idxs, 0)
        return res

    def set(self, key, data):
        ntype = self.graph.ntypes[self.ntype_idx]
        return self.graph.set_ndata(key=key, data=data, ntype=ntype)

class EdgeDataView(object):
    def __init__(self, graph, etype_idx, idxs=None):
        self.graph = graph
        self.etype_idx = etype_idx
        self.idxs = idxs

    def __getitem__(self, key):
        res = self.graph.edge_frames[self.etype_idx][key]
        if self.idxs is not None:
            # res = res[self.idxs]
            res = jnp.take(res, self.idxs, 0)
        return res

    def set(self, key, data):
        assert self.idxs is None, "Cannot partially set. "
        etype = self.graph.etypes[self.etype_idx]
        return self.graph.set_edata(key=key, data=data, etype=etype)
