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
        self.typeidx = typeidx
        self.short_name = short_name
        self.long_name = long_name
        self.get_number = lambda idx: getattr(
            self.graph.gidx, "number_of_%ss" % long_name,
        )(idx)
        self.idxs = idxs

    def __getitem__(self, key: str):
        res = getattr(
            self.graph, "%s_frames" % self.long_name)[self.typeidx][key]
        if self.idxs is not None:
            res = res[self.idxs]
        return res

    def set(self, key: str, data: jnp.ndarray):
        assert self.idxs is None, "Cannot partially set. "
        assert data.shape[0] == self.get_number(self.typeidx)
        frame = getattr(
            self.graph, "%s_frames" % self.long_name)[self.typeidx]
        if frame is None:
            frame = {}
        frame = unfreeze(frame)
        frame[key] = data
        frame = freeze(frame)
        frames = getattr(
            self.graph, "%s_frames" % self.long_name)[:self.typeidx]\
            + (frame, )\
            + getattr(
                self.graph, "%s_frames" % self.long_name)[self.typeidx+1:]
        graph = replace(
            self.graph,
            **{
                "%s_frames" % self.long_name: frames
            }
        )
        return graph

from functools import partial
NodeView = partial(EntityView, short_name="n", long_name="node")
EdgeView = partial(EntityView, short_name="e", long_name="edge")
NodeDataView = partial(DataView, short_name="n", long_name="node")
EdgeDataView = partial(DataView, short_name="e", long_name="edge")
