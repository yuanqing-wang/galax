"""Implementation for core graph computation."""
from .heterograph import HeteroGraph
from typing import Callable, Optional
from flax.core import freeze, unfreeze
from dataclasses import replace
from functools import partial
from . import function
from .function import ReduceFunction
import jax
from jax.tree_util import tree_map

def message_passing(
        graph: HeteroGraph,
        mfunc: Optional[Callable],
        rfunc: Optional[ReduceFunction],
        afunc: Optional[Callable]=None,
        etype: Optional[Callable]=None,
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
    >>> mfunc = lambda edge: {"m": edge.srcdata["h"]}
    >>> rfunc = ReduceFunction("sum", "m", "h1")
    >>> _g = message_passing(g, mfunc, rfunc)
    >>> _g.ndata['h1'].flatten().tolist()
    [0.0, 1.0, 1.0]

    """

    # TODO(yuanqing-wang): change this restriction in near future
    assert isinstance(rfunc, ReduceFunction), "Only built-in reduce supported. "

    # find the edge type
    etype_idx = graph.get_etype_id(etype)

    # extract the message
    message = mfunc(graph.edges[etype])

    # reduce by calling jax.ops.segment_
    _rfunc = getattr(function, "segment_%s" % rfunc.op)
    _rfunc = partial(
        _rfunc,
        segment_ids=graph.gidx.edges[0][1],
        num_segments=graph.number_of_nodes()
    )
    reduced = {rfunc.out_field: _rfunc(message[rfunc.msg_field])}

    # apply if so specified
    if afunc is not None:
        reduced.update(afunc(reduced))

    # update destination node frames
    srctype_idx, dsttype_idx = graph.gidx.metagraph.find_edge(etype_idx)
    node_frame = graph.node_frames[dsttype_idx]
    node_frame = unfreeze(node_frame)
    node_frame.update(reduced)
    node_frame = freeze(node_frame)
    node_frames = graph.node_frames[:dsttype_idx] + (node_frame, )\
        + graph.node_frames[dsttype_idx+1:]

    return replace(graph, node_frames=node_frames)