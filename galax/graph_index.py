"""Module for graph index class definition."""
from typing import Any, NamedTuple, Iterable, Mapping, Union, Optional
import jax.numpy as jnp
import numpy as onp
from jax.experimental.sparse import BCOO

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
        return spmat

    def incidence_matrix(self, typestr):
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
        typestr : str
            Can be either "in", "out" or "both"

        Returns
        -------
        SparseTensor
            The incidence matrix.
        jnp.ndarray
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        src, dst, eid = self.edges()
        n = self.number_of_nodes()
        m = self.number_of_edges()
        if typestr == 'in':
            row, col = dst, eid
            idx = jnp.stack([row, col], axis=0)
            dat = jnp.ones((m,))
            inc = BCOO(data=dat, indices=idx, shape=(n, m))
        elif typestr == 'out':
            row, col = src, eid
            idx = jnp.stack([row, col], axis=0)
            dat = jnp.ones((m,))
            inc = BCOO(data=dat, indices=idx, shape=(n, m))
        elif typestr == 'both':
            # first remove entries for self loops
            mask = (src != dst)
            src = src[mask]
            dst = dst[mask]
            eid = eid[mask]
            n_entries = src.shape[0]
            # create index
            row = jnp.concatenate([src, dst], axis=0)
            col = jnp.concatenate([eid, eid], axis=0)
            idx = jnp.stack([row, col], axis=0)
            # FIXME(minjie): data type
            x = -jnp.ones((n_entries,))
            y = jnp.ones((n_entries,))
            dat = jnp.concatenate([x, y], axis=0)
            inc = BCOO(data=dat, indices=idx, shape=(n, m))
        return inc

    def to_networkx(self):
        """Convert to networkx graph.

        The edge id will be saved as the 'id' edge attribute.

        Returns
        -------
        networkx.DiGraph
            The nx graph
        """
        src, dst, eid = self.edges()
        # xiangsx: Always treat graph as multigraph
        ret = nx.MultiDiGraph()
        ret.add_nodes_from(range(self.number_of_nodes()))
        for u, v, e in zip(src, dst, eid):
            ret.add_edge(u, v, id=e)
        return ret

def from_coo(num_nodes, src, dst):
    """Convert from coo arrays.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    src : Tensor
        Src end nodes of the edges.
    dst : Tensor
        Dst end nodes of the edges.

    Returns
    -------
    GraphIndex
        The graph index.
    """
    return GraphIndex(
        n_nodes=num_nodes,
        src=src,
        dst=dst,
    )

from networkx(nx_graph):
    """Convert from networkx graph.

    If 'id' edge attribute exists, the edge will be added follows
    the edge id order. Otherwise, order is undefined.

    Parameters
    ----------
    nx_graph : networkx.DiGraph
        The nx graph or any graph that can be converted to nx.DiGraph

    Returns
    -------
    GraphIndex
        The graph index.
    """
    if not isinstance(nx_graph, nx.Graph):
        nx_graph = nx.DiGraph(nx_graph)
    else:
        if not nx_graph.is_directed():
            # to_directed creates a deep copy of the networkx graph even if
            # the original graph is already directed and we do not want to do it.
            nx_graph = nx_graph.to_directed()
    num_nodes = nx_graph.number_of_nodes()

    # nx_graph.edges(data=True) returns src, dst, attr_dict
    if nx_graph.number_of_edges() > 0:
        has_edge_id = 'id' in next(iter(nx_graph.edges(data=True)))[-1]
    else:
        has_edge_id = False

    if has_edge_id:
        num_edges = nx_graph.number_of_edges()
        src = np.zeros((num_edges,), dtype=np.int64)
        dst = np.zeros((num_edges,), dtype=np.int64)
        for u, v, attr in nx_graph.edges(data=True):
            eid = attr['id']
            src[eid] = u
            dst[eid] = v
    else:
        src = []
        dst = []
        for e in nx_graph.edges:
            src.append(e[0])
            dst.append(e[1])
    num_nodes = nx_graph.number_of_nodes()
    # We store edge Ids as an edge attribute.
    src = jnp.array(src)
    dst = jnp.array(dst)
    return from_coo(num_nodes, src, dst)

def from_scipy_sparse_matrix(adj):
    """Convert from scipy sparse matrix.

    Parameters
    ----------
    adj : scipy sparse matrix

    Returns
    -------
    GraphIndex
        The graph index.
    """
    num_nodes = max(adj.shape[0], adj.shape[1])
    adj_coo = adj.tocoo()
    return from_coo(num_nodes, adj_coo.row, adj_coo.col)

def from_edge_list(elist, readonly):
    """Convert from an edge list.

    Parameters
    ---------
    elist : list, tuple
        List of (u, v) edge tuple, or a tuple of src/dst lists

    """
    if isinstance(elist, tuple):
        src, dst = elist
    else:
        src, dst = zip(*elist)
    src_ids = jnp.asarray(src)
    dst_ids = jnp.asarray(dst)
    num_nodes = max(src.max(), dst.max()) + 1
    return from_coo(num_nodes, src_ids, dst_ids)

def create_graph_index(graph_data):
    """Create a graph index object.

    Parameters
    ----------
    graph_data : graph data
        Data to initialize graph. Same as networkx's semantics.

    """
    if isinstance(graph_data, GraphIndex):
        # FIXME(minjie): this return is not correct for mutable graph index
        return graph_data

    if graph_data is None:
        return GraphIndex()
    elif isinstance(graph_data, (list, tuple)):
        # edge list
        return from_edge_list(graph_data, readonly)
    elif isinstance(graph_data, scipy.sparse.spmatrix):
        # scipy format
        return from_scipy_sparse_matrix(graph_data, readonly)
    else:
        try:
            gidx = from_networkx(graph_data, readonly)
        except Exception:
            raise RuntimeError(
                "'Error while creating graph from input of type "%s".'
                % type(graph_data)"
            )
        return gidx
