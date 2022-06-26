"""Utilities for batching graphs."""
from typing import Sequence, Union
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
    >>> import galax
    >>> g = galax.graph(([0, 0, 2], [0, 1, 2]))
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

    >>> _g.gdata["_batched_num_nodes"].flatten().tolist()
    [3, 3]

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
    batched_num_nodes = n_nodes
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

    # (n_graphs, n_ntypes)
    original_batched_num_nodes = [
        graph.graph_frame.get("_batched_num_nodes", default=None)
        if graph.graph_frame is not None
        else None
        for graph in graphs
    ]

    batched_num_nodes = [
        jnp.expand_dims(batched_num_nodes[idx], 0)
        if original_batched_num_nodes[idx] is None
        else original_batched_num_nodes[idx]
        for idx in range(len(original_batched_num_nodes))
    ]

    batched_num_nodes = jnp.concatenate(batched_num_nodes)

    if graphs[0].graph_frame is not None:
        if list(graphs[0].graph_frame.keys()) != ["_batched_num_nodes"]:
            graph_frame = {
                    key:
                    jnp.concatenate(
                        [graph.graph_frame[key] for graph in graphs]
                    )
                    for key in graphs[0].graph_frame.keys()
                    if key != "_batched_num_nodes"
            }
        else:
            graph_frame = {}
    else:
        graph_frame = {}

    graph_frame.update({"_batched_num_nodes": batched_num_nodes})

    return HeteroGraph.init(
        gidx=gidx,
        ntypes=ntypes,
        etypes=etypes,
        node_frames=node_frames,
        edge_frames=edge_frames,
        graph_frame=graph_frame,
    )

def pad(
        graphs: Union[Sequence[HeteroGraph], HeteroGraph],
        n_nodes: Union[int, jnp.ndarray],
        n_edges: Union[int, jnp.ndarray],
):
    """Pad graphs to desired number of nodes and edges and batch them.

    Parameters
    ----------
    graphs : Union[Sequence[HeteroGraph], HeteroGraph]
        A sequence of graphs, could be already batched.
    n_nodes : Union[int, jnp.ndarray]
        Number of nodes.
    n_edges : Union[int, jnp.ndarray]
        Number of edges.

    Returns
    -------
    HeteroGraph
        Batched graph with desired padding.

    Examples
    --------
    >>> import galax
    >>> g = galax.graph(([0, 0, 2], [0, 1, 2]))
    >>> _g = pad(g, g.number_of_nodes(), g.number_of_edges())
    >>> _g.gidx == g.gidx
    True

    >>> _g = pad(g, 5, 8)
    >>> int(_g.number_of_edges())
    8
    >>> int(_g.number_of_nodes())
    5

    >>> g = g.ndata.set("h", jnp.ones(3))
    >>> g = g.edata.set("h", jnp.ones(3))
    >>> _g = pad(g, 5, 8)
    >>> _g.ndata["h"].tolist()
    [1.0, 1.0, 1.0, 0.0, 0.0]
    >>> _g.edata["h"].tolist()
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> _g.gdata["_batched_num_nodes"].flatten().tolist()
    [3, 2]
    >>> bool(_g.gdata["_has_dummy"])
    True

    """
    if not isinstance(graphs, HeteroGraph):
        graphs = batch(graphs)
    current_n_nodes = graphs.gidx.n_nodes
    current_n_edges = jnp.array([len(edge[0]) for edge in graphs.gidx.edges])
    delta_n_nodes = n_nodes - current_n_nodes
    delta_n_edges = n_edges - current_n_edges
    n_nodes = delta_n_nodes
    edges = tuple(
        [
            (jnp.zeros(_n_edge), jnp.zeros(_n_edge))
            for _n_edge in delta_n_edges
        ]
    )
    gidx = HeteroGraphIndex(
        metagraph=graphs.gidx.metagraph,
        n_nodes=n_nodes,
        edges=edges,
    )

    node_frames = (
        FrozenDict(
            {
                key:
                jnp.zeros(
                    (n_nodes[idx], *graphs.node_frames[idx][key].shape[1:]),
                )
                for key in graphs.node_frames[idx].keys()
            }
        )
        if graphs.node_frames[idx] is not None else None
        for idx in range(len(graphs.node_frames))
    )

    edge_frames = (
        FrozenDict(
            {
                key:
                jnp.zeros(
                    (
                        delta_n_edges[idx],
                        *graphs.edge_frames[idx][key].shape[1:],
                    ),
                )
                for key in graphs.edge_frames[idx].keys()
            }
        )
        if graphs.edge_frames[idx] is not None else None
        for idx in range(len(graphs.edge_frames))
    )

    dummy = HeteroGraph.init(
        gidx=gidx,
        ntypes=graphs.ntypes,
        etypes=graphs.etypes,
        node_frames=node_frames,
        edge_frames=edge_frames,
    )

    g = batch([graphs, dummy])
    g = g.gdata.set("_has_dummy", jnp.array(True))

    return g
