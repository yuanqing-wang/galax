import abc
from typing import Callable, Optional
from flax import linen as nn
from ..function import apply_nodes, apply_edges

class Module(nn.Module):
    @abc.abstractmethod
    def uno(self, *args, **kwargs):
        raise NotImplementedError

    def init(fn, *args, **kwargs):
        if "method" not in kwargs:
            kwargs["method"] = self.uno
        return super().init(*args, **kwargs)


class Sequential(nn.Sequential):
    def uno(self, graph, field="h"):
        h = graph.ndata[field]
        for layer in self.layers:
            h = layer.uno(h)
        return h

class ApplyNodes(Module):
    layer: Callable

    def uno(self, h):
        return self.layer(h)

    def __call__(self, graph):
        return apply_nodes(self.uno)(graph)
