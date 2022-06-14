"""Built-in functions."""
from typing import Optional
from collections import namedtuple
import jax
import jax.numpy as jnp

ReduceFunction = namedtuple(
    "ReduceFunction",
    ["op", "msg_field", "out_field"]
)

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
