"""Implementation for core graph computation."""
from .heterograph import HeteroGraph
from typing import Callable
from flax.core import freeze, unfreeze
from dataclasses import replace
from .function import ReduceFunction
from jax.tree_util import tree_map

def message_passing(
        g: HeteroGraph,
        mfunc: Optional[Callable],
        rfunc: Optional[ReduceFunction],
        afunc: Optional[Callable],
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

    """
    assert len(g.ntypes) == 1, "Only one ntype supported. "
    assert isinstance(rfunc, ReduceFunction), "Only built-in reduce supported. "
    message = mfunc(g.edges)
    _rfunc = getattr(jax.ops, "segmentation_%s" % rfunc.op)
    _rfunc = lambda _message:  _rfunc(
        _message,
        segment_ids=g.gidx.edges[0].dst,
        num_segments=g.number_of_nodes(),
    )
    reduced = tree_map(_rfunc, message)
    if afunc is not None:
        reduced = tree_map(afunc, reduced)
    node_frames = self.node_frames[0]
    node_frames = unfreeze(node_frames)
    node_frames.update(reduced)
    node_frames = freeze(node_frames)
    node_frames = (node_frames, )

    return replace(g, node_frames=node_frames)
