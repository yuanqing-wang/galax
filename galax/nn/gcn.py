from typing import Optional
from flax import linen as nn


class GCN(nn.Module):
    """Graph convolutional layer from
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
    """

    pass
