"""Views of Graph.
Inspired by dgl.view
"""

from collections import namedtuple
from dataclasses import replace
from typing import Optional
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

DataSpace = namedtuple("DataSpace", ["data"])

class EntityView(object):
    def __init__(self, graph, short_name: str, long_name: str):
        self.graph = graph
        self.short_name = short_name
        self.long_name = long_name
        self.get_id = lambda key: getattr(
            self.graph, "get_%stype_id" % short_name
        )(key)
        self.get_number = lambda idx: getattr(
            self.graph, "number_of_%ss" % long_name,
        )(idx)

    def __getitem__(self, key: str):
        typeidx = self.get_id(key)
        return DataSpace(
            data=DataView(
                graph=self.graph,
                typeidx=typeidx,
                short_name=self.short_name,
                long_name=self.long_name,
            ),
        )

    def __call__(self, typestr: Optional[str]=None):
        return jnp.arange(self.get_number(typestr))

class DataView(object):
    def __init__(self, graph, typeidx: int, short_name: str, long_name: str):
        self.graph = graph
        self.typeidx = typeidx
        self.short_name = short_name
        self.long_name = long_name
        self.get_number = lambda idx: getattr(
            self.graph.gidx, "number_of_%ss" % long_name,
        )(idx)

    def __getitem__(self, key: str):
        return getattr(
            self.graph, "%s_frames" % self.long_name)[self.typeidx][key]

    def set(self, key: str, data: jnp.ndarray):
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
