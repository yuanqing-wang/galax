"""Built-in functions."""
import sys
from typing import Optional, Callable
from functools import partial
from itertools import product
from collections import namedtuple
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn

# =============================================================================
# MESSAGE FUNCTIONS
# =============================================================================
CODE2STR = {
    "u": "source",
    "v": "destination",
    "e": "edge",
}

CODE2OP = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "dot": lambda x, y: (x * y).sum(axis=-1, keepdims=True),
}

CODE2DATA = {
    "u": "src",
    "e": "data",
    "v": "dst",
}


def copy_u(u, out):
    """Builtin message function that computes message using source node
    feature.

    Parameters
    ----------
    u : str
        The source feature field.
    out : str
        The output message field.
    """
    return lambda edge: {out: edge.src[u]}


def copy_e(e, out):
    """Builtin message function that computes message using edge feature.

    Parameters
    ----------
    e : str
        The edge feature field.
    out : str
        The output message field.
    """
    return lambda edge: {out: edge.data[e]}


def _gen_message_builtin(lhs, rhs, binary_op):
    name = "{}_{}_{}".format(lhs, binary_op, rhs)
    docstring = """Builtin message function that computes a message on an edge
    by performing element-wise {} between features of {} and {}
    if the features have the same shape; otherwise, it first broadcasts
    the features to a new shape and performs the element-wise operation.

    Broadcasting follows NumPy semantics. Please see
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    for more details about the NumPy broadcasting semantics.

    Parameters
    ----------
    lhs_field : str
        The feature field of {}.
    rhs_field : str
        The feature field of {}.
    out : str
        The output message field.

    """.format(
        binary_op,
        CODE2STR[lhs],
        CODE2STR[rhs],
        CODE2STR[lhs],
        CODE2STR[rhs],
    )

    # grab data field
    lhs_data, rhs_data = CODE2DATA[lhs], CODE2DATA[rhs]

    # define function
    def func(lhs_field, rhs_field, out):
        def fn(edges):
            return {out:
                    CODE2OP[binary_op](
                        getattr(edges, lhs_data)[lhs_field],
                        getattr(edges, rhs_data)[rhs_field],
                        )
                    }
        return fn

    # attach name and doc
    func.__name__ = name
    func.__doc__ = docstring
    return func


def _register_builtin_message_func():
    """Register builtin message functions"""
    target = ["u", "v", "e"]
    for lhs, rhs in product(target, target):
        if lhs != rhs:
            for binary_op in ["add", "sub", "mul", "div", "dot"]:
                func = _gen_message_builtin(lhs, rhs, binary_op)
                setattr(sys.modules[__name__], func.__name__, func)


_register_builtin_message_func()


# =============================================================================
# REDUCE FUNCTIONS
# =============================================================================

ReduceFunction = namedtuple(
    "ReduceFunction", ["op", "msg_field", "out_field"]
)

sum = partial(ReduceFunction, "sum")
mean = partial(ReduceFunction, "mean")
max = partial(ReduceFunction, "max")
min = partial(ReduceFunction, "min")

segment_sum = jax.ops.segment_sum

def segment_max(*args, **kwargs):
    """Alias of jax.ops.segment_max with nan_to_num."""
    return jnp.nan_to_num(
        jax.ops.segment_max(*args, **kwargs),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

def segment_min(*args, **kwargs):
    """Alias of jax.ops.segment_min with nan_to_num."""
    return jnp.nan_to_num(
        jax.ops.segment_min(*args, **kwargs),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

def segment_mean(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
):
    """Returns mean for each segment.

    Reference
    ---------
    * Shamelessly stolen from jraph.utils

    Parameters
    ----------
    data : jnp.ndarray
        the values which are averaged segment-wise.
    segment_ids : jnp.ndarray
        indices for the segments.
    num_segments : Optional[int]
        total number of segments.
    indices_are_sorted : bool=False
        whether ``segment_ids`` is known to be sorted.
    unique_indices : bool=False
        whether ``segment_ids`` is known to be free of duplicates.

    Returns
    -------
    jnp.ndarray
        The data after segmentation sum.
    """
    nominator = segment_sum(
        data,
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    denominator = segment_sum(
        jnp.ones_like(data),
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    return nominator / jnp.maximum(
        denominator, jnp.ones(shape=[], dtype=denominator.dtype)
    )


# =============================================================================
# APPLY FUNCTIONS
# =============================================================================
def apply_nodes(
    function: Callable,
    in_field: str = "h",
    out_field: Optional[str] = None,
    ntype: Optional[str] = None,
):
    """Apply a function to node attributes.

    Parameters
    ----------
    function : Callable
        Input function.
    in_field : str
        Input field
    out_field : str
        Output field.

    Returns
    -------
    Callable
        Function that takes and returns a graph.

    Examples
    --------
    Transform function.
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import galax
    >>> graph = galax.graph(((0, 1), (1, 2)))
    >>> graph = graph.ndata.set("h", jnp.ones(3))
    >>> fn = apply_nodes(lambda x: x * 2)
    >>> graph = jax.jit(fn)(graph)
    >>> graph.ndata['h'].tolist()
    [2.0, 2.0, 2.0]

    """
    if out_field is None:
        out_field = in_field

    def _fn(graph, in_field=in_field, out_field=out_field, ntype=ntype):
        ntype_idx = graph.get_ntype_id(ntype)
        node_frame = unfreeze(graph.node_frames[ntype_idx])
        node_frame[out_field] = function(node_frame[in_field])
        node_frame = freeze(node_frame)
        node_frames = graph.node_frames[:ntype_idx] + (node_frame,)\
            + graph.node_frames[ntype_idx+1:]
        return graph._replace(node_frames=node_frames)

    return _fn

def apply_edges(
    function: Callable,
    in_field: str = "h",
    out_field: Optional[str] = None,
    etype: Optional[str] = None,
):
    """Apply a function to edge attributes.

    Parameters
    ----------
    function : Callable
        Input function.
    in_field : str
        Input field
    out_field : str
        Output field.

    Returns
    -------
    Callable
        Function that takes and returns a graph.

    Examples
    --------
    Transform function.
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import galax
    >>> graph = galax.graph(((0, 1), (1, 2)))
    >>> graph = graph.edata.set("h", jnp.ones(2))
    >>> fn = apply_edges(lambda x: x * 3)
    >>> graph = jax.jit(fn)(graph)
    >>> graph.edata['h'].tolist()
    [3.0, 3.0]

    """
    if out_field is None:
        out_field = in_field

    def _fn(graph, in_field=in_field, out_field=out_field, etype=etype):
        etype_idx = graph.get_etype_id(etype)
        edge_frame = unfreeze(graph.edge_frames[etype_idx])
        edge_frame[out_field] = function(edge_frame[in_field])
        edge_frame = freeze(edge_frame)
        edge_frames = graph.edge_frames[:etype_idx] + (edge_frame, )\
            + graph.edge_frames[etype_idx+1:]
        return graph._replace(edge_frames=edge_frames)

    return _fn
