"""Module for heterogeneous graph index class definition."""
from typing import Any, NamedTuple, Iterable, Mapping, Union, Optional, Tuple
from .graph_index import GraphIndex

class HeteroGraphIndex(NamedTuple):
    """HeteroGraph index object.

    """
    metagraph: GraphIndex
    n_nodes: jnp.ndarray # (number of nodes per ntype)
    edges: Tuple[Tuple[jnp.ndarray]] # within each edge type, src and dst

    def number_of_ntypes(self):
        """Return the number of node types.

        Returns
        -------
        int
            The number of node types.
        """
        return self.metagraph.number_of_nodes()

    def number_of_etypes(self):
        """Return the number of edge types.

        Returns
        -------
        int
            The number of edge types.
        """
        return self.metagraph.number_of_edges()

    def add_nodes(self, ntype: int, num: int):
        """Add nodes.

        Parameters
        ----------
        ntype : int
            Node type
        num : int
            Number of nodes to be added.
        """
        if ntype >= len(self.n_nodes):
            metagraph = self.metagraph.add_nodes(1)
            n_nodes = jnp.concatenate([self.n_nodes, num])
            edges = self.edges

        else:
            metagraph = self.metagraph
            n_nodes = self.n_nodes.at[ntype].add(num)
            edges = self.edges

        return self.__class__(
            metagraph=metagraph, n_nodes=n_nodes, edges=edges,
        )

    def add_edge(
            self, etype: int, src: int, dst: int,
            srctype: int, dsttype: int,
        ):
        """Add one edge.

        Parameters
        ----------
        etype : int
            Edge type
        src : int
            The src node.
        dst : int
            The dst node.
        srctype: int
            The src node type.
        dsttype: int
            The dst node type.

        """
