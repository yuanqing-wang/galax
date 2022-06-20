"""Graph Convolutional network.<https://arxiv.org/abs/1609.02907>`__"""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
from ... import function as fn

class GCN(nn.Module):
    r"""Graph convolutional layer from
    `Semi-Supervised Classification with Graph Convolutional
    Networks <https://arxiv.org/abs/1609.02907>`__

    Mathematically it is defined as follows:

    .. math::
      h_i^{(l+1)} =
      \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}
      \frac{1}{c_{ji}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`c_{ji}` is the product of the square root of node degrees
    (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`),
    and :math:`\sigma` is an activation function.

    Parameters
    ----------
    out_feats : int
        Output features size.
    norm : Optional[str]

    Returns
    -------
    HeteroGraph
        The resulting Graph.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import galax
    >>> g = galax.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = g.add_self_loop()
    >>> g = g.set_ndata("h", jnp.ones((6, 10)))
    >>> gcn = GCN(2, use_bias=True)
    >>> params = gcn.init(jax.random.PRNGKey(2666), g)
    >>> g = gcn.apply(params, g)
    >>> x = g.ndata['h']
    >>> x.shape
    (6, 2)
    """
    features: int
    use_bias: bool = False
    activation: Optional[Callable] = jax.nn.relu

    @nn.compact
    def uno(self, h, norm=1.0):
        # initialize parameters
        kernel = self.param(
            'kernel',
            jax.nn.initializers.glorot_uniform(),
            (h.shape[-1], self.features),
        )

        if self.use_bias:
            bias = self.param(
                "bias",
                jax.nn.initializers.zeros,
                (self.features, ),
            )
        else:
            bias = 0.0

        activation = self.activation
        if activation is None:
            activation = lambda x: x
        return activation((h @ kernel) * norm + bias)

    def __call__(self, graph, field="h"):
        # propergate
        graph = graph.update_all(fn.copy_u(field, "m"), fn.sum("m", field))

        # normalize
        degrees = graph.out_degrees()
        norm = degrees ** (-0.5)
        norm_shape = norm.shape + (1, ) * (graph.ndata[field].ndim - 1)
        norm = jnp.reshape(norm, norm_shape)

        # transform
        function = lambda h: self.uno(h, norm=norm)
        graph = fn.apply_nodes(function, field)(graph)
        return graph
