"""Built-in functions."""
import sys
from typing import Optional
from functools import partial
from itertools import product
from collections import namedtuple
import jax
import jax.numpy as jnp

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
    "u": "srcdata",
    "e": "data",
    "v": "dstdata",
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
    return lambda edge: {out: edge.srcdata[u]}

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

    """.format(binary_op,
               CODE2STR[lhs],
               CODE2STR[rhs],
               CODE2STR[lhs],
               CODE2STR[rhs],
    )

    # grab data field
    lhs_data, rhs_data = CODE2DATA[lhs], CODE2DATA[rhs]

    # define function
    func = lambda edge: CODE2OP[binary_op](
        getattr(edge, lhs_data), getattr(edge, rhs_data),
    )

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
    "ReduceFunction",
    ["op", "msg_field", "out_field"]
)

sum = partial(ReduceFunction, "sum")
mean = partial(ReduceFunction, "mean")
max = partial(ReduceFunction, "max")
min = partial(ReduceFunction, "min")

segment_sum = jax.ops.segment_sum
segment_max = jax.ops.segment_max
segment_min = jax.ops.segment_min

def segment_mean(data: jnp.ndarray,
                 segment_ids: jnp.ndarray,
                 num_segments: Optional[int] = None,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False):
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
        unique_indices=unique_indices)
    denominator = segment_sum(
        jnp.ones_like(data),
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices)
    return nominator / jnp.maximum(denominator,
                                 jnp.ones(shape=[], dtype=denominator.dtype))
