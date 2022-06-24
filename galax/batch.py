"""Utilities for batching graphs."""
from typing import Sequence
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from .heterograph import HeteroGraph
from .heterograph_index import HeteroGraphIndex


def batch(graphs: Sequence[HeteroGraph]):
    """Batch a sequence of graphs into one.

    Parameters
    ----------
    graphs : Sequence[HeteroGraph]
        Sequence of graphs.

    Returns
    -------
    HeteroGraph
        The batched graph.

    Examples
    --------
    >>> g = graph(([0, 0, 2], [0, 1, 2]))
    >>> _g = batch([g, g])
    >>> _g.gidx.edges[0][0].tolist()
    [0, 0, 2, 3, 3, 5]
    >>> _g.gidx.edges[0][1].tolist()
    [0, 1, 2, 3, 4, 5]

    >>> g0 = g.ndata.set("h", jnp.zeros(3))
    >>> g1 = g.ndata.set("h", jnp.ones(3))
    >>> _g = batch([g0, g1])
    >>> _g.ndata["h"].tolist()
    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    """
    # make sure the metagraphs are exactly the same
    assert all(graph.ntypes == graphs[0].ntypes for graph in graphs)
    assert all(graph.etypes == graphs[0].etypes for graph in graphs)
    assert all(
        graph.gidx.metagraph == graphs[0].gidx.metagraph
        for graph in graphs
    )
    metagraph = graphs[0].gidx.metagraph

    # ntypes and etypes remain the same
    etypes = graphs[0].etypes
    ntypes = graphs[0].ntypes

    # number of nodes on offsets
    n_nodes = jnp.stack([graph.gidx.n_nodes for graph in graphs])
    offsets = jnp.cumsum(n_nodes[:-1], axis=0)
    offsets = jnp.concatenate(
        [jnp.zeros((1, offsets.shape[-1]), dtype=jnp.int32), offsets]
    )
    n_nodes = n_nodes.sum(axis=0)

    # edge indices with offsets added
    num_edge_types = len(graphs[0].gidx.edges)
    edges = [[[], []] for _ in range(num_edge_types)]
    for idx_etype in range(num_edge_types):
        for idx_graph, graph in enumerate(graphs):
            src, dst = graph.gidx.edges[idx_etype]
            src = src + offsets[idx_graph]
            dst = dst + offsets[idx_graph]
            edges[idx_etype][0].append(src)
            edges[idx_etype][1].append(dst)
        edges[idx_etype][0] = jnp.concatenate(edges[idx_etype][0])
        edges[idx_etype][1] = jnp.concatenate(edges[idx_etype][1])
        edges[idx_etype] = tuple(edges[idx_etype])
    edges = tuple(edges)
    gidx = HeteroGraphIndex(n_nodes=n_nodes, edges=edges, metagraph=metagraph)

    # concatenate frames
    node_frames = (
        FrozenDict(
            {
                key:
                jnp.concatenate(
                    [graph.node_frames[idx][key] for graph in graphs]
                )
                for key in graphs[0].node_frames[idx].keys()
            }
        )
        if graphs[0].node_frames[idx] is not None else None
        for idx in range(len(graphs[0].node_frames))
    )

    edge_frames = (
        FrozenDict(
            {
                key:
                jnp.concatenate(
                    [graph.edge_frames[idx][key] for graph in graphs]
                )
                for key in graphs[0].edge_frames[idx].keys()
            }
        )
        if graphs[0].edge_frames[idx] is not None else None
        for idx in range(len(graphs[0].edge_frames))
    )

    return HeteroGraph.init(
        gidx=gidx,
        ntypes=ntypes,
        etypes=etypes,
        node_frames=node_frames,
        edge_frames=edge_frames
    )
