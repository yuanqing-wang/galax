import abc
from typing import Callable, Optional
from flax import linen as nn
from ..function import apply_nodes, apply_edges

class ApplyNodes(nn.Module):
    layer: Callable

    def uno(self, h):
        return self.layer(h)

    def __call__(self, graph):
        return apply_nodes(self.uno)(graph)
