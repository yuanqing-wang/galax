"""`GraphSAGE <https://cs.stanford.edu/~jure/pubs/graphsage-nips17.pdf>`__"""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
from ... import function as fn

class GraphSAGE(nn.Module):
    r"""GraphSAGE layer from `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__
    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} &= \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)
        h_{i}^{(l+1)} &= \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)
        h_{i}^{(l+1)} &= \mathrm{norm}(h_{i}^{(l+1)})

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        If aggregator type is ``gcn``, the feature size of source and
        destination nodes are required to be the same.
    out_feats : int
        Output feature size; i.e, the number of dimensions
        of :math:`h_i^{(l+1)}`.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    feat_drop : float
        Dropout rate on features, default: ``0``.
    use_bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    features: int
    aggregator_type: str = "mean"
    use_bias: bool = True
    activation: Optional[Callable] = None

    @nn.compact
    def __call__(self, graph, field="h"):
        h_self = graph.ndata[field]

        if self.aggregator_type == "mean":
            graph = graph.update_all(
                fn.copy_src(field, "m"), fn.mean("m", "neigh"),
            )
            h_neigh = graph.ndata["neigh"]

        elif self.aggregator_type == "gcn":
            graph = graph.update_all(
                fn.copy_src(field, "m"), fn.sum("m", "neigh"),
            )
            degrees = graph.in_degrees()
            h_neigh = (graph.ndata["neigh"] + graph.ndata[field])\
                / (jnp.expand_dims(degrees, -1) + 1)

        elif self.aggregator_type == "pool":
            h_pool = jax.nn.relu(
                nn.Dense(graph.ndata[field].shape[-1])(graph.ndata[field]),
            )
            graph = graph.ndata.set(field, h_pool)
            graph = graph.update_all(
                fn.copy_src(field, "m"), fn.max("m", "neigh"),
            )
            h_neigh = graph.ndata["neigh"]

        h_neigh = nn.Dense(self.features, use_bias=False)(h_neigh)

        if self.aggregator_type == "gcn":
            rst = h_neigh
        else:
            rst = h_neigh + nn.Dense(self.features, use_bias=False)(h_self)

        if self.activation is not None:
            rst = self.activation(rst)

        return rst
