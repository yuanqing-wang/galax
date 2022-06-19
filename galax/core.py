"""Implementation for core graph computation."""
from typing import Callable, Optional, Any
from functools import partial
from flax.core import freeze, unfreeze
from . import function
from .function import ReduceFunction


def message_passing(
    graph: Any,
    mfunc: Optional[Callable],
    rfunc: Optional[ReduceFunction],
    afunc: Optional[Callable] = None,
    etype: Optional[Callable] = None,
):
    """Invoke message passing computation on the whole graph.

    Parameters
    ----------
    g : HeteroGraph
        The input graph.
    mfunc : Callable
        Message function.
    rfunc : Callable
        Reduce function.
    afunc : Callable
        Apply function.

    Returns
    -------
    HeteroGraph
        The resulting graph.

    Examples
    --------
    >>> import galax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> g = galax.graph(((0, 1), (1, 2)))
    >>> g = g.ndata.set("h", jnp.ones(3))
    >>> mfunc = galax.function.copy_u("h", "m")
    >>> rfunc = galax.function.sum("m", "h1")
    >>> _g = message_passing(g, mfunc, rfunc)
    >>> _g.ndata['h1'].flatten().tolist()
    [0.0, 1.0, 1.0]

    """
    # TODO(yuanqing-wang): change this restriction in near future
    # assert isinstance(rfunc, ReduceFunction), "Only built-in reduce supported. "
    if etype is None:
        etype = graph.etypes[0]

    # find the edge type
    etype_idx = graph.get_etype_id(etype)

    # get number of nodes
    _, dsttype_idx = graph.get_meta_edge(etype_idx)
    n_dst = next(iter(graph.node_frames[dsttype_idx].values())).shape[0]

    # extract the message
    message = mfunc(graph.edges[etype])

    # reduce by calling jax.ops.segment_
    _rfunc = getattr(function, f"segment_{rfunc.op}")
    _rfunc = partial(
        _rfunc,
        segment_ids=graph.gidx.edges[0][1],
        num_segments=n_dst,
    )
    reduced = {rfunc.out_field: _rfunc(message[rfunc.msg_field])}

    # apply if so specified
    if afunc is not None:
        reduced.update(afunc(reduced))

    # update destination node frames
    node_frame = graph.node_frames[dsttype_idx]
    node_frame = unfreeze(node_frame)
    node_frame.update(reduced)
    node_frame = freeze(node_frame)
    node_frames = (
        graph.node_frames[:dsttype_idx]
        + (node_frame,)
        + graph.node_frames[dsttype_idx + 1 :]
    )

    return graph._replace(node_frames=node_frames)
