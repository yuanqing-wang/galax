"""The data loader function for multiple graphs."""

import random
from typing import Sequence
import jax
import jax.numpy as jnp
from ..heterograph import HeteroGraph
from ..batch import pad

class PrixFixeDataLoader:
    """A helper object that shuffles and iterates over graphs.

    Parameters
    ----------
    graphs : Sequence[HeteroGraph]
        Graphs to iterate over.
    batch_size : int = 1
        Batch size.

    Examples
    --------
    >>> import galax
    >>> g0 = galax.graph(((0, 1), (1, 2)))
    >>> g1 = galax.graph(((0, 1, 2), (1, 2, 3)))
    >>> g2 = galax.graph(((0, 1, 2, 3), (1, 2, 3, 4)))
    >>> dataloader = PrixFixeDataLoader((g0, g1, g2), batch_size=3)
    >>> dataloader.max_num_edges.item()
    9
    >>> dataloader.max_num_nodes.item()
    12
    >>> g = next(iter(dataloader))
    >>> int(g.number_of_nodes())
    12
    >>> int(g.number_of_edges())
    9


    """
    def __init__(
        self,
        graphs: Sequence[HeteroGraph],
        batch_size: int = 1,
    ):
        self.graphs = graphs
        self.batch_size = batch_size
        self._graphs = None
        self._prepare()

    def _prepare(self):
        """Compute the max nodes and max edges for padding and batching."""
        # compute max n_nodes and n_edges
        # (n_graphs, n_ntypes)
        n_nodes = jnp.stack(
            [graph.gidx.n_nodes for graph in self.graphs],
            axis=0,
        )

        # (n_graphs, n_etypes)
        n_edges = jnp.stack(
            [
                jnp.array([len(edge[0]) for edge in graph.gidx.edges])
                for graph in self.graphs
            ]
        )

        # (k, n_ntypes)
        top_n_nodes = jax.lax.top_k(n_nodes.T, self.batch_size)[0].T

        # (k, n_etypes)
        top_n_edges = jax.lax.top_k(n_edges.T, self.batch_size)[0].T

        max_n_nodes = top_n_nodes.sum(0)
        max_n_edges = top_n_edges.sum(0)

        self.max_num_nodes = max_n_nodes
        self.max_num_edges = max_n_edges

    def __iter__(self):
        self._graphs = list(self.graphs)
        random.shuffle(self._graphs)
        return self

    def __next__(self):
        if len(self._graphs) < self.batch_size:
            raise StopIteration
        else:
            graphs_to_serve = self._graphs[:self.batch_size]
            self._graphs = self._graphs[self.batch_size:]
            graphs_to_serve = pad(
                graphs_to_serve,
                self.max_num_nodes,
                self.max_num_edges,
            )
            return graphs_to_serve
