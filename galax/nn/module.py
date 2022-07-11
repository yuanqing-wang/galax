import abc
from typing import Callable, Optional
from flax import linen as nn
from ..function import apply_nodes, apply_edges

class ApplyNodes(nn.Module):
    layer: Callable

    def __call__(self, graph, field="h"):
        graph = graph.ndata.set(field, self.layer(graph.ndata[field]))
        return graph
