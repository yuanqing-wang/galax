"""Module for graph index class definition."""
from typing import Any, NamedTuple, Iterable, Mapping, Union, Optional
import jax.numpy as jnp
import numpy as onp

class GraphIndex(NamedTuple):
    """Graph index object.

    Attriubtes
    ----------
    n_nodes : int
        The number of nodes in the graph.
    src : jnp.ndarray
        The indices of the source nodes, for each edge.
    dst : jnp.ndarray
        The indices of the destination nodes, for each edge.

    Note
    ----
    All transformations returns new GraphIndex object rather than modify it
    in-place.

    """
    n_nodes: int
    src: Optional[jnp.ndarray]
    dst: Optional[jnp.ndarray]

    def add_nodes(self, num):
        """Add nodes.

        Parameters
        ----------
        num : int
            Number of nodes to be added.
        """
        return self.__class__(
            n_nodes=self.n_nodes+num,
            src=self.src,
            dst=self.dst,
        )

    def add_edge(self, u, v):
        """Add one edge.
        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.
        """
        return self.__class__(
            n_nodes=self.n_nodes,
            src=jnp.concatenate([src, jnp.array([u])]),
            dst=jnp.concatenate([dst, jnp.array([v])]),
        )

    def is_multigraph(self):
        """Return whether the graph is a multigraph

        Returns
        -------
        bool
            True if it is a multigraph, False otherwise.
        """
        src_and_dst = jnp.stack([self.src, self.dst], axis=-1)
        return jnp.unique(src_and_dst, axis=0).shape[0] == src_and_dst.shape[0]

    def number_of_nodes(self):
        """Return the number of nodes.

        Returns
        -------
        int
            The number of nodes
        """
        return self.n_nodes

    def number_of_edges(self):
        """Return the number of edges.

        Returns
        -------
        int
            The number of edges
        """
        return self.src.shape[0]

    def has_node(self, vid):
        """Return true if the node exists.

        Parameters
        ----------
        vid : int
            The nodes

        Returns
        -------
        bool
            True if the node exists, False otherwise.
        """
        return vid < self.number_of_nodes()

    def has_nodes(self, vids):
        """Return true if the nodes exist.

        Parameters
        ----------
        vid : jnp.ndarray
            The nodes

        Returns
        -------
        jnp.ndarray
            0-1 array indicating existence
        """
        return 1 * (vid < self.number_of_nodes)

    def has_edge_between(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        bool
            True if the edge exists, False otherwise
        """
        u_in_src = (u == self.src)
        v_in_dst = (v == self.dst)
        return (u_in_src * v_in_dst).any()

    def has_edges_between(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : jnp.ndarray
            The src nodes.
        v : jnp.ndarray
            The dst nodes.

        Returns
        -------
        jnp.ndarray
            0-1 array indicating existence
        """
        result = []
        for _u, _v in zip(u, v):
            result.append(int(self.has_edge_between(_u, _v)))
        return jnp.array(result)

    def eid(self, u, v):
        """Return the id array of all edges between u and v.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        jnp.ndarray
            The edge id array.
        """
        u_in_src = (u == self.src)
        v_in_dst = (v == self.dst)
        return jnp.where(u_in_src * v_in_dst)

    def eids(self, u, v):
        """Return a triplet of arrays that contains the edge IDs.

        Parameters
        ----------
        u : jnp.ndarray
            The src nodes.
        v : jnp.ndarray
            The dst nodes.

        Returns
        -------
        jnp.ndarray
            The src nodes.
        jnp.ndarray
            The dst nodes.
        jnp.ndarray
            The edge ids.
        """
        eid = self.eid(u, v)
        n_edges = len(eid)
        src = jnp.array([u for _ in range(n_edges)])
        dst = jnp.array([v for _ in range(n_edges)])
        return src, dst, eid

    def find_edge(self, eid):
        """Return the edge tuple of the given id.

        Parameters
        ----------
        eid : int
            The edge id.

        Returns
        -------
        int
            src node id
        int
            dst node id
        """
        return self.src[eid], self.dst[eid]

    def find_edges(self, eid):
        """Return a triplet of arrays that contains the edge IDs.

        Parameters
        ----------
        eid : jnp.ndarray
            The edge ids.

        Returns
        -------
        jnp.ndarray
            The src nodes.
        jnp.ndarray
            The dst nodes.
        jnp.ndarray
            The edge ids.
        """
        src = []
        dst = []
        for _eid in eid:
            _src, _dst = self.find_edge(_eid)
            src.append(_src)
            dst.append(_dst)
        src = jnp.array(src)
        dst = jnp.array(dst)
        return src, dst, eid

    def in_edges(self, v):
        """Return the in edges of the node(s).

        Parameters
        ----------
        v : jnp.ndarray
            The node(s).

        Returns
        -------
        jnp.ndarray
            The src nodes.
        jnp.ndarray
            The dst nodes.
        jnp.ndarray
            The edge ids.
        """
        v = jnp.expand_dims(v, -1)
        dst = jnp.expand_dims(self.dst, 0)

        # (len(v), len(dst))
        v_is_dst = (v == dst)
        return jnp.where(v_is_dst, axis=-1)

    def out_edges(self, v):
        """Return the out edges of the node(s).

        Parameters
        ----------
        v : jnp.ndarray
            The node(s).

        Returns
        -------
        jnp.ndarray
            The src nodes.
        jnp.ndarray
            The dst nodes.
        jnp.ndarray
            The edge ids.
        """
        v = jnp.expand_dims(v, -1)
        src = jnp.expand_dims(self.src, 0)

        # (len(v), len(src))
        v_is_src = (v == src)
        return jnp.where(v_is_src, axis=-1)

    def edges(self, order=None):
       """Return all the edges.

        Parameters
        ----------
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
        src, dst, eid = self.src, self.dst, jnp.arange(len(self.src))
        idxs = jnp.lexsort((src, dst))
        if order == "srcdst":
            src, dst, eid = src[idxs], dst[idxs], eid[idxs]
        return src, dst, eid

    def in_degree(self, v):
        """Return the in degree of the node.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        int
            The in degree.
        """
        return (v == self.dst).sum()

    def in_degrees(self, v):
        """Return the in degrees of the nodes.

        Parameters
        ----------
        v : jnp.ndarray
            The nodes.

        Returns
        -------
        tensor
            The in degree array.
        """
        v = jnp.expand_dims(v, -1)
        dst = jnp.expand_dims(self.dst, 0)

        # (len(v), len(dst))
        v_is_dst = (v == dst)

        return v_is_dst.sum(axis=-1)

    def out_degree(self, v):
        """Return the out degree of the node.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        int
            The out degree.
        """
        return (v == self.src).sum()

    def out_degrees(self, v):
        """Return the out degrees of the nodes.

        Parameters
        ----------
        v : jnp.ndarray
            The nodes.

        Returns
        -------
        tensor
            The out degree array.
        """
        v = jnp.expand_dims(v, -1)
        src = jnp.expand_dims(self.src, 0)

        # (len(v), len(src))
        v_is_dst = (v == src)

        return v_is_dst.sum(axis=-1)

    def adjacency_matrix_scipy(self, transpose, fmt, return_edge_ids=None):
        """Return the scipy adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the
        destination of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column
        represents a destination.

        Parameters
        ----------
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
        if return_edge_ids is None:
            return_edge_ids = True

        if fmt is not "coo":
            raise NotImplementedError
        else:
            n = self.number_of_nodes()
            m = self.number_of_edges()
            if transpose:
                row, col = onp.array(self.src), onp.array(self.dst)
            else:
                row, col = onp.array(self.dst), onp.array(self.src)
            data = onp.arange(0, m) if return_edge_ids else onp.ones_like(row)
            return scipy.sparse.coo_matrix((data, (row, col)), shape=(n, n))

    def adjacency_matrix(self, transpose, ctx):
        """Return the adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        transpose : bool
            A flag to transpose the returned adjacency matrix.
        ctx : context
            The context of the returned matrix.

        Returns
        -------
        SparseTensor
            The adjacency matrix.
        jnp.ndarray
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        from jax.experimental.sparse import BCOO
        m = self.number_of_edges()
        n = self.number_of_nodes()
        if transpose:
            row, col = onp.array(self.src), onp.array(self.dst)
        else:
            row, col = onp.array(self.dst), onp.array(self.src)
        idx = jnp.stack([row, col], axis=-1)
        data = jnp.ones((m,))
        shape = (n, n)
        spmat = BCOO(data=data, indices=idx, shape=shape)
        return spmat, None
