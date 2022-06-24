"""`Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__"""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
from ... import function as fn
from ..module import Module

class GAT(Module):
    r"""
    Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.
    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:
    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})
        e_{ij}^{l} &=
        \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    features : int
        Features
    num_heads : int
        Number of attention heads.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import galax
    >>> g = galax.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = g.add_self_loop()
    >>> g = g.set_ndata("h", jnp.ones((6, 10)))
    >>> gat = GAT(2, 4, deterministic=True)
    >>> params = gat.init(jax.random.PRNGKey(2666), g.ndata['h'])
    >>> g = gat.apply(params, g)
    >>> x = g.ndata['h']
    >>> x.shape
    (6, 4, 2)
    """

    features: int
    num_heads: int
    feat_drop: Optional[float] = 0.0
    attn_drop: Optional[float] = 0.0
    negative_slope: float = 0.2
    activation: Optional[Callable] = None
    deterministic: bool = True
    use_bias: bool = True

    def setup(self):
        self.fc = nn.Dense(
            self.features * self.num_heads, use_bias=False,
            kernel_init=nn.initializers.variance_scaling(
                3.0, "fan_avg", "uniform"
            ),
        )

        self.attn_l = nn.Dense(
            1,
            kernel_init=nn.initializers.variance_scaling(
                3.0, "fan_avg", "uniform"
            ),
        )

        self.attn_r = nn.Dense(
            1,
            kernel_init=nn.initializers.variance_scaling(
                3.0, "fan_avg", "uniform"
            ),
        )

        if self.use_bias:
            self.bias = self.param(
                "bias",
                nn.zeros,
                (self.num_heads, self.features),
            )

        self.dropout_feat = nn.Dropout(self.feat_drop, deterministic=self.deterministic)
        self.dropout_attn = nn.Dropout(self.attn_drop, deterministic=self.deterministic)

    def uno(self, h):
        h = self.dropout_feat(h)
        h = self.fc(h)
        h = h.reshape(h.shape[:-1] + (self.num_heads, self.features))
        el = self.attn_l(h)
        er = self.attn_r(h)
        e = el + er
        e = self.dropout_attn(e)
        return h

    def __call__(self, graph, field="h", etype="E_"):
        h = graph.ndata[field]
        h0 = h
        h = self.dropout_feat(h)
        h = self.fc(h)
        h = h.reshape(h.shape[:-1] + (self.num_heads, self.features))
        el = self.attn_l(h)
        er = self.attn_r(h)
        graph = graph.ndata.set(field, h)
        graph = graph.ndata.set("er", er)
        graph = graph.ndata.set("el", el)
        e = graph.edges[etype].src["er"] + graph.edges[etype].dst["el"]
        e = nn.leaky_relu(e, self.negative_slope)
        a = fn.segment_softmax(e, graph.edges[etype].dst.idxs, graph.number_of_nodes())
        a = self.dropout_attn(a)
        graph = graph.edata.set("a", a)
        graph = graph.update_all(
            fn.u_mul_e(field, "a", "m"),
            fn.sum("m", field)
        )

        if self.use_bias:
            graph = fn.apply_nodes(
                lambda x: x + self.bias, in_field=field, out_field=field
            )(graph)

        if self.activation is not None:
            graph = fn.apply_nodes(
                self.activation, in_field=field, out_field=field
            )(graph)
        return graph
