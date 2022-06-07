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

    def add_edges(
            self, etype: int, src: jnp.ndarray, dst: jnp.ndarray,
            srctype: Optional[int], dsttype: Optional[int],
        ):
        """And many edges.

        Parameters
        ----------
        etype : int
            Edge type
        src : jnp.ndarray
            The src node.
        dst : jnp.ndarray
            The dst node.
        srctype: Optional[int]
            The src node type. Necessary if etype is new.
        dsttype: Optional[int]
            The dst node type. Necessary if etype is new.

        """
        if etype < len(self.edges):
            assert srctype is None and dsttype is None
            metagraph = self.metagraph
            edges = self.edges[:etype] +\
                (
                    jnp.concatenate([self.edges[etype][0], src]),
                    jnp.concatenate([self.edges[etype][1], dst]),
                ) + self.edges[(etype)+1:]

        else:
            assert etype == len(self.edges), "Edges are sorted. "
            assert srctype is not None and dsttype is not None
            metagraph = self.metagraph.add_edge(srctype, dsttype)
            edges = edges + (src, dst)

        return self.__class__(
            metagraph=metagraph, n_nodes=self.n_nodes, edges=edges,
        )

    def add_edge(
            self, etype: int, src: int, dst: int,
            srctype: Optional[int], dsttype: Optional[int],
    ):
        """And one edge.

        Parameters
        ----------
        etype : int
            Edge type
        src : int
            The src node.
        dst : int
            The dst node.
        srctype: Optional[int]
            The src node type. Necessary if etype is new.
        dsttype: Optional[int]
            The dst node type. Necessary if etype is new.

        """
        src = jnp.array([src])
        dst = jnp.array([dst])
        return self.add_edges(
            etype=etype, src=src, dst=dst, src_type=srctype, dsttype=dsttype,
        )

    def etype_subgraph(self, etype: int):
        """Create a unitgraph graph from given edge type.

        Parameters
        ----------
        etype : int
            Edge type.

        Returns
        -------
        GraphIndex
        """
        srctype, dsttype = self.metagraph.find_edge(etype)
        if srctype == dsttype:
            n_nodes = self.n_nodes[srctype]
            src, dst = self.edges[etype]
        else:
            n_nodes = self.n_nodes[srctype] + self.n_nodes[dsttype]
            src, dst = self.edges[etype]
            dst = dst + self.n_nodes[srctype]

        return GraphIndex(
            n_nodes=n_nodes, src=src, dst=dst,
        )

    def is_multigraph(self):
        """Return whether the graph is a multigraph

        Returns
        -------
        bool
            True if it is a multigraph, False otherwise.
        """
        return any(
            [
                self.etype_subgraph(etype).is_multigraph()
                for etype in range(len(self.edges))
            ]
        )

    def number_of_nodes(self, ntype: int) -> int:
        """Return the number of nodes.

        Parameters
        ----------
        ntype : int
            Node type

        Returns
        -------
        int
            The number of nodes
        """
        return self.n_nodes[ntype]

    def number_of_edges(self, etype: int) -> int:
        """Return the number of edges.

        Parameters
        ----------
        etype : int
            Edge type

        Returns
        -------
        int
            The number of edges
        """
        return len(self.edges[etype])

    def has_nodes(self, ntype: int, vids: jnp.ndarray):
        """Return true if the nodes exist.

        Parameters
        ----------
        ntype : int
            Node type
        vid : jnp.ndarray
            Node IDs

        Returns
        -------
        jnp.ndarray
            0-1 array indicating existence
        """
        return 1 * (vids < self.n_nodes[ntype])

    def has_edges_between(self, etype: int, u: int, v: int) -> bool:
        """Return true if the edge exists.

        Parameters
        ----------
        etype : int
            Edge type
        u : jnp.ndarray
            Src node Ids.
        v : jnp.ndarray
            Dst node Ids.

        Returns
        -------
        jnp.ndarray
            0-1 array indicating existence
        """
        src, dst = self.edges[etype]

        u = jnp.expand_dims(u, -1)
        v = jnp.expand_dims(v, -1)

        src = jnp.expand_dims(src, -1)
        dst = jnp.expand_dims(dst, -1)

        return 1 * ((u == src) * (v == dst)).any(axis=-1)

    def find_edges(self, etype: int, eid: jnp.ndarray) -> Tuple[jnp.ndarray]:
        """Return a triplet of arrays that contains the edge IDs.

        Parameters
        ----------
        etype : int
            Edge type
        eid : jnp.ndarray
            Edge ids.

        Returns
        -------
        jnp.ndarray
            The src nodes.
        jnp.ndarray
            The dst nodes.
        jnp.ndarray
            The edge ids.
        """
        edge = self.edges[etype]
        src, dst = edge
        return src[eid], dst[eid], eid

    def edges(self, etype: int, order: Optional[str]=None):
        """Return all the edges

        Parameters
        ----------
        etype : int
            Edge type
        order : string
            The order of the returned edges. Currently support:
            - 'srcdst' : sorted by their src and dst ids.
            - 'eid'    : sorted by edge Ids.
            - None     : the arbitrary order.

        Returns
        -------
        jnp.ndarray
            The src nodes.
        jnp.ndarray
            The dst nodes.
        jnp.ndarray
            The edge ids.
        """
        src, dst = self.edges[etype]
        src, dst, eid = src, dst, jnp.arange(len(self.src))
        if order == "srcdst":
            idxs = jnp.lexsort((src, dst))
            src, dst, eid = src[idxs], dst[idxs], eid[idxs]
        return src, dst, eid

    def in_degrees(self, etype: int, v: jnp.ndarray):
        """Return the in degrees of the nodes.

        Assume that node_type(v) == dst_type(etype).
        Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : jnp.ndarray
            The nodes.

        Returns
        -------
        jnp.ndarray
            The in degree array.
        """
        v = jnp.expand_dims(v, -1)
        _, dst = self.edges[etype]
        dst = jnp.expand_dims(dst, 0)

        # (len(v), len(dst))
        v_is_dst = (v == dst)

        return v_is_dst.sum(axis=-1)


    def out_degrees(self, etype, v):
        """Return the out degrees of the nodes.

        Assume that node_type(v) == src_type(etype).
        Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : jnp.ndarray
            The nodes.

        Returns
        -------
        jnp.ndarray
            The out degree array.
        """
        v = jnp.expand_dims(v, -1)
        src, _ = self.edges[etype]
        src = jnp.expand_dims(src, 0)

        # (len(v), len(dst))
        v_is_src = (v == src)

        return v_is_src.sum(axis=-1)

    def adjacency_matrix(
            self,
            etype: int,
            transpose: bool=False,
        ) -> BCOO:
        """Return the adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents
        the source of an edge and the column represents the destination.

        When transpose is True, a row represents the destination and
        a column represents the source.

        Parameters
        ----------
        etype : int
            Edge type
        transpose : bool
            A flag to transpose the returned adjacency matrix.
        ctx : context
            The context of the returned matrix.

        Returns
        -------
        BCOO
            The adjacency matrix.
        jnp.ndarray
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        return self.etype_subgraph(etype).adjacency_matrix(transpose=transpose)

    def adjacency_matrix_scipy(
            self,
            etype: int,
            transpose: bool=False,
            fmt: str="coo",
            return_edge_ids: Optional[bool]=None,
        ):
        """Return the scipy adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents
        the destination of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column
        represents a destination.

        Parameters
        ----------
        etype : int
            Edge type
        transpose : bool
            A flag to transpose the returned adjacency matrix.
        fmt : str
            Indicates the format of returned adjacency matrix.
        return_edge_ids : bool
            Indicates whether to return edge IDs or 1 as elements.

        Returns
        -------
        scipy.sparse.spmatrix
            The scipy representation of adjacency matrix.
        """
        return self.etype_subgraph(etype).adjacency_matrix_scipy(
            transpose=transpose, fmt=fmt, return_edge_ids=return_edge_ids
        )

    def incidence_matrix(
        self,
        etype: int,
        typestr: str,
    ):
        """Return the incidence matrix representation of this graph.
        An incidence matrix is an n x m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.

        There are three types of an incidence matrix `I`:
        * "in":
          - I[v, e] = 1 if e is the in-edge of v (or v is the dst node of e);
          - I[v, e] = 0 otherwise.
        * "out":
          - I[v, e] = 1 if e is the out-edge of v (or v is the src node of e);
          - I[v, e] = 0 otherwise.
        * "both":
          - I[v, e] = 1 if e is the in-edge of v;
          - I[v, e] = -1 if e is the out-edge of v;
          - I[v, e] = 0 otherwise (including self-loop).

        Parameters
        ----------
        etype : int
            Edge type
        typestr : str
            Can be either "in", "out" or "both"
        ctx : context
            The context of returned incidence matrix.

        Returns
        -------
        BCOO
            The incidence matrix.
        jnp.ndarray
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        return self.etype_subgraph(etype).incidence_matrix(typestr=typestr)

    def reverse(self):
        """Reverse the heterogeneous graph adjacency
        The node types and edge types are not changed.

        Returns
        -------
        HeteroGraphIndex
            A new graph index.
        """
        return self.__class__(
            n_nodes=self.n_nodes,
            edges=tuple((dst, src) for src, dst in self.edges),
        )
