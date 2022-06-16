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

class EntityView(object):
    def __init__(self, graph, short_name: str, long_name: str):
        self.graph = graph
        self.short_name = short_name
        self.long_name = long_name
        get_id = lambda key: getattr(
            self.graph, "get_%stype_id" % short_name
        )(key)
        get_number = lambda idx: getattr(
            self.graph, "number_of_%ss" % long_name,
        )(idx)

        get_id = jax.jit(get_id, static_argnums=(0, ))
        self.get_id = get_id
        self.get_number = get_number

    def __getitem__(self, key: str):
        typeidx = self.get_id(key)
        if self.short_name == "n":
            return NodeSpace(
                data=DataView(
                    graph=self.graph,
                    typeidx=typeidx,
                    short_name="n",
                    long_name="node",
                ),
            )
        elif self.short_name == "e":
            srctype_idx, dsttype_idx = self.graph.gidx.metagraph.find_edge(
                typeidx,
            )

            src, dst = self.graph.gidx.edges[typeidx]

            return EdgeSpace(
                data=DataView(
                    graph=self.graph,
                    typeidx=typeidx,
                    short_name="e",
                    long_name="edge",
                ),
                srcdata=DataView(
                    graph=self.graph,
                    typeidx=srctype_idx,
                    short_name="n",
                    long_name="node",
                    idxs=src,
                ),
                dstdata=DataView(
                    graph=self.graph,
                    typeidx=dsttype_idx,
                    short_name="n",
                    long_name="node",
                    idxs=dst,
                ),
            )

    def __call__(self, typestr: Optional[str]=None):
        return jnp.arange(self.get_number(typestr))

class DataView(object):
    def __init__(
            self,
            graph,
            typeidx: str,
            short_name: str,
            long_name: str,
            idxs: Optional[jnp.ndarray]=None,
        ):
        self.graph = graph
        self.typeidx = str(typeidx)
        self.short_name = short_name
        self.long_name = long_name
        self.idxs = idxs

    def __getitem__(self, key: str):
        res = getattr(
            self.graph, "%s_frames" % self.long_name)[int(self.typeidx)][key]
        if self.idxs is not None:
            res = res[self.idxs]
        return res

    def set(self, key: str, data: jnp.ndarray):
        assert self.idxs is None, "Cannot partially set. "
        frame = getattr(
            self.graph, "%s_frames" % self.long_name)[int(self.typeidx)]
        if frame is None:
            frame = {}
        frame = unfreeze(frame)
        frame[key] = data
        frame = freeze(frame)
        frames = getattr(
            self.graph, "%s_frames" % self.long_name)[:int(self.typeidx)]\
            + (frame, )\
            + getattr(
                self.graph, "%s_frames" % self.long_name)[int(self.typeidx)+1:]
        graph = replace(
            self.graph,
            **{
                "%s_frames" % self.long_name: frames
            }
        )
        return graph

from functools import partial
NodeDataView = partial(DataView, short_name="n", long_name="node")
EdgeDataView = partial(DataView, short_name="e", long_name="edge")

def prefetch_view(graph):
    nodes = {}
    edges = {}
    for idx, ntype in enumerate(graph.ntypes):
        nodes[ntype] = NodeSpace(
            data=DataView(
                graph=graph,
                typeidx=idx,
                short_name="n",
                long_name="node",
            ),
        )

    for idx, etype in enumerate(graph.etypes):
        if len(graph.gidx.metagraph.src) == 0:
            continue
        srctype_idx, dsttype_idx = graph.gidx.metagraph.find_edge(idx)
        src, dst = graph.gidx.edges[idx]
        edges[etype] = EdgeSpace(
            data=DataView(
                graph=graph,
                typeidx=idx,
                short_name="e",
                long_name="edge",
            ),
            srcdata=DataView(
                graph=graph,
                typeidx=srctype_idx,
                short_name="n",
                long_name="node",
                idxs=src,
            ),
            dstdata=DataView(
                graph=graph,
                typeidx=dsttype_idx,
                short_name="n",
                long_name="node",
                idxs=dst,
            ),
        )

    return nodes, edges
