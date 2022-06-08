from typing import Any, NamedTuple, Iterable, Mapping, Union, Optional, Tuple
from .heterograph_index import HeteroGraphIndex
from flax.core import FrozenDict

class HeteroGraph(NamedTuple):
    """Class for storing graph structure and node/edge feature data.

    Parameters
    ----------
    gidx : HeteroGraphIndex
        Graph index object.
    ntypes : list of str
        Node type list. ``ntypes[i]`` stores the name of node type i.
        If a pair is given, the graph created is a
        uni-directional bipartite graph,
        and its SRC node types and DST node types are given as in the pair.
    etypes : list of str
        Edge type list. ``etypes[i]`` stores the name of edge type i.
    node_frames : list[Frame], optional
        Node feature storage. If None, empty frame is created.
        Otherwise, ``node_frames[i]`` stores the node features
        of node type i. (default: None)
    edge_frames : list[Frame], optional
        Edge feature storage. If None, empty frame is created.
        Otherwise, ``edge_frames[i]`` stores the edge features
        of edge type i. (default: None)

    """
    gidx: Optional[HeteroGraphIndex]=None
    ntypes: Optional[Tuple]=None
    etypes: Optional[Tuple]=None
    node_frames: Optional[Tuple]=None
    edge_frames: Optional[Tuple]=None

    if gidx is None: gidx = []
    if ntypes is None: ntypes = ["_N"]
    if etypes is None: etypes = ["_E"]

    _ntype_invmap = FrozenDict({idx: ntype for idx, ntype in enumerate(ntypes)})
    _etype_invmap = FrozenDict({idx: etype for idx, etype in enumerate(etypes)})

    def add_nodes(
            self,
            num: int,
            data: Optional[Dict]=None,
            ntype: Optional[Dict]=None,
        ):
        """Add new nodes of the same node type

        Parameters
        ----------
        num : int
            Number of nodes to add.
        data : dict, optional
            Feature data of the added nodes.
        ntype : str, optional
            The type of the new nodes. Can be omitted if there is
            only one node type in the graph.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import galax

        **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.num_nodes()
        3
        >>> g.add_nodes(2)
        >>> g.num_nodes()
        5

        If the graph has some node features and new nodes are added without
        features, their features will be created by initializers defined
        with :func:`set_n_initializer`.

        >>> g.ndata['h'] = torch.ones(5, 1)
        >>> g.add_nodes(1)
        >>> g.ndata['h']
        tensor([[1.], [1.], [1.], [1.], [1.], [0.]])

        We can also assign features for the new nodes in adding new nodes.

        >>> g.add_nodes(1, {'h': torch.ones(1, 1), 'w': torch.ones(1, 1)})
        >>> g.ndata['h']
        tensor([[1.], [1.], [1.], [1.], [1.], [0.], [1.]])

        Since ``data`` contains new feature fields, the features for old nodes
        will be created by initializers defined with :func:`set_n_initializer`.

        >>> g.ndata['w']
        tensor([[0.], [0.], [0.], [0.], [0.], [0.], [1.]])


        **Heterogeneous Graphs with Multiple Node Types**

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        ...                                 torch.tensor([0, 0, 1, 1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        ...                                         torch.tensor([0, 1]))
        ...     })
        >>> g.add_nodes(2)
        DGLError: Node type name must be specified
        if there are more than one node types.
        >>> g.num_nodes('user')
        3
        >>> g.add_nodes(2, ntype='user')
        >>> g.num_nodes('user')
        5

        See Also
        --------
        remove_nodes
        add_edges
        remove_edges
        """
        if ntype is None:
            assert len(self.ntypes) == 1, "Please specify node types. "
            ntype = self.ntypes[0]

        if ntype not in self.ntypes:
            gidx = self.gidx.add_nodes(ntype=len(self.ntypes), num=num)
            ntypes = self.ntypes + (ntype, )
            etypes = self.etypes
            if data is not None:
                data = FrozenDict(data)
                node_frames = self.node_frames + (data, )
            else:
                node_frames = self.node_frames + (None, )
            edge_frames = self.edge_frames

        else:
            ntype_idx = self._ntype_invmap(ntype)
            gidx = self.gidx.add_nodes(ntype=self.ntype_idx, num=num)
            ntypes = self.ntypes
            etypes = self.etypes
            if data is not None:
                data = FrozenDict(data)
                original_data = self.node_frames[ntype]
                assert list(original_data.keys()) == list(data.keys())
                new_data = FrozenDict(
                    {
                        key: jnp.concatenate(
                            [
                                original_data[key],
                                data[key],
                            ]
                            for key in original_data.keys(),
                        )
                    }
                )
                node_frames = self.node_frames[:ntype_idx]\
                    + (new_data, ) + self.node_frames[ntype_idx+1:]
            else:
                node_frames = self.node_frames
            edge_frames = self.edge_frames
            return self.__class__(
                gidx=gidx, ntypes=ntypes, etypes=etypes,
                node_frames=node_frames, edge_frames=edge_frames,
            )

    def add_edges(
            self,
            u: jnp.ndarray, v: jnp.ndarray,
            data: Optional[jnp.ndarray]=None, etype: Optional[str]=None,
            srctype: Optional[str]=None, dsttype: Optional[str]=None,
        ):
        if etype is None:
            assert len(self.etypes) == 1, "Etype needs to be specified. "
            etype = self.etypes[0]

        ntypes = self.ntypes
        node_frames = self.node_frames

        if etype in self._etype_invmap:
            etype_idx = self._etype_invmap[etype]
            gidx = self.gidx.add_edges(etype=etype_idx, src=u, dst=v)

            if data is not None:
                data = FrozenDict(data)
                original_data = self.edge_frames[etype]
                assert list(original_data.keys()) == list(data.keys())
                new_data = FrozenDict(
                    {
                        key: jnp.concatenate(
                            [
                                original_data[key],
                                data[key],
                            ]
                            for key in original_data.keys(),
                        )
                    }
                )
                edge_frames = self.edge_frames[:etype_idx]\
                    + (new_data, ) + self.edge_frames[etype_idx+1:]
                etype = self.etype
            else:
                gidx = self.gidx.add_edges(
                    etype=etype_idx, src=u, dst=v,
                    srctype=self._ntype_invmap[srctype],
                    dsttype=self._ntype_invmap[dsttype],
                )
                etypes = self.etypes + (etype, )
                edge_frames = self.edge_frames + (FrozenDict(data), )

        return self.__class__(
            gidx=gidx, ntypes=ntypes, etypes=etypes,
            node_frames=node_frames, edge_frames=edge_frames,
        )

    def remove_edges(self, eids: jnp.array, etype: Optional[str]=None):
        etype_idx = self.get_etype_id(etype)

        # handle the data corresponding to the edges
        sub_edge_frame = self.edge_frames[etype_idx]
        sub_edge_frame = FrozenDict(
            {
                key: jnp.delete(value, eids)
                for key, value in sub_edge_frame.items()
            }
        )

        example_key, example_value = next(sub_edge_frame.items())
        if len(example_value) == 0:
            edge_frames =\
                self.edge_frames[:etype_idx] + self.edge_frames[etype_idx+1:]
            etypes =\
                self.etypes[:etype_idx] + self.etypes[etype_idx+1:]
        else:
            edge_frames = self.edge_frames[:etype_idx]\
                + sub_edge_frame + self.edge_frames[etype_idx+1:]
            etypes = self.etypes

        gidx = self.gidx.remove_edges(etype=etype_idx, eids=eids)
        return self.__class__(
            gidx=gidx,
            ntypes=self.ntypes,
            etypes=etypes,
            node_frames=self.node_frames,
            edge_frames=edge_frames,
        )

    def remove_nodes(self, nids: jnp.ndarray, ntype: Optional[str]=None):
        ntype_idx = self.get_ntype_id(ntype)
        gidx = self.gidx.remove_nodes(ntype=ntype_idx, nids=nids)

        # handle the data corresponding to the edges
        sub_node_frame = self.edge_frames[ntype_idx]
        sub_node_frame = FrozenDict(
            {
                key: jnp.delete(value, eids)
                for key, value in sub_node_frame.items()
            }
        )

        example_key, example_value = next(sub_node_frame.items())
        if len(example_value) == 0:
            node_frames =\
                self.node_frames[:ntype_idx] + self.node_frames[ntype_idx+1:]
            ntypes =\
                self.ntypes[:ntype_idx] + self.ntypes[ntype_idx+1:]
        else:
            node_frames = self.node_frames[:ntype_idx]\
                + sub_edge_frame + self.node_frames[ntype_idx+1:]
            ntypes = self.ntypes

        return self.__class__(
            gidx=gidx,
            ntypes=ntypes,
            etypes=self.etypes,
            node_frames=node_frames,
            edge_frames=self.edge_frames,
        )

    def canonical_etypes(self):
        """Return all the canonical edge types in the graph.

        A canonical edge type is a string triplet ``(str, str, str)``
        for source node type, edge type and destination node type.

        Returns
        -------
        list[(str, str, str)]
            All the canonical edge type triplets in a list.

        Notes
        -----
        DGL internally assigns an integer ID for each edge type. The returned
        edge type names are sorted according to their IDs.

        See Also
        --------
        etypes

        Examples
        --------
        The following example uses PyTorch backend.
        >>> import dgl
        >>> import torch
        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        ... })
        >>> g.canonical_etypes
        [('user', 'follows', 'user'),
         ('user', 'follows', 'game'),
         ('user', 'plays', 'game')]
        """
        return [
            (
                self.ntypes[self.gidx.metagraph.src[etype_idx]],
                self.etypes[etype_idx],
                self.ntypes[self.gidx.metagraph.dst[etype_idx]],
            )
            for etype_idx in range(len(self.etypes))
        ]

    def to_canonincal_etype(self, etype: str) -> Tuple[str]:
        """Convert an edge type to the corresponding canonical edge type in the graph.

        A canonical edge type is a string triplet ``(str, str, str)``
        for source node type, edge type and destination node type.
        The function expects the given edge type name can uniquely identify
        a canonical edge type.

        Parameters
        ----------
        etype : str
            If :attr:`etype` is an edge type (str), it returns the corresponding canonical edge
            type in the graph.

        Returns
        -------
        (str, str, str)
            The canonical edge type corresponding to the edge type.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> import dgl
        >>> import torch
        Create a heterograph.
        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        ...     ('developer', 'follows', 'game'): ([0, 1], [0, 1])
        ... })
        Map an edge type to its corresponding canonical edge type.
        >>> g.to_canonical_etype('plays')
        ('user', 'plays', 'game')
        >>> g.to_canonical_etype(('user', 'plays', 'game'))
        ('user', 'plays', 'game')
        See Also
        --------
        canonical_etypes
        """
        etype_idx = self.get_etype_id(etype)
        src, dst = self.gidx.metagraph.find_edge(etype_idx)
        return self.ntypes[src], etype, self.ntyes[dst]

    def get_ntype_id(self, ntype: Optional[str]=None) -> int:
        """Return the ID of the given node type.
        ntype can also be None. If so, there should be only one node type in the
        graph.

        Parameters
        ----------
        ntype : str
            Node type

        Returns
        -------
        int
        """
        if ntype is None:
            assert len(self.ntypes) == 1, "Ntype needs to be specified. "
            return 0
        else:
            assert ntype in self._ntype_invmap, "No such ntype. "
            return self._ntype_invmap[ntype]

    def get_etype_id(self, etype: Optional[str]=None) -> int:
        """Return the id of the given edge type.
        etype can also be None. If so, there should be only one edge type in the
        graph.

        Parameters
        ----------
        etype : str or tuple of str
            Edge type

        Returns
        -------
        int
        """
        if etype is None:
            assert len(self.etypes) == 1, "Etype needs to be specified. "
            return 0
        else:
            assert ntype in self._etype_invmap, "No such etype. "
            return self._etype_invmap[etype]

    def number_of_nodes(self, ntype: Optional[str]=None):
        return self.gidx.number_of_nodes(self.get_ntype_id(ntype))

    def number_of_edges(self, etype: Optional[str]=None):
        return self.gidx.number_of_edges(self.get_etype_id(etype))

    def is_multigraph(self):
        """Return whether the graph is a multigraph with parallel edges.
        A multigraph has more than one edges between the same pair of nodes, called
        *parallel edges*.  For heterogeneous graphs, parallel edge further requires
        the canonical edge type to be the same (see :meth:`canonical_etypes` for the
        definition).

        Returns
        -------
        bool
            True if the graph is a multigraph.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> import dgl
        >>> import torch
        Check for homogeneous graphs.
        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 3])))
        >>> g.is_multigraph
        False
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 3, 3])))
        >>> g.is_multigraph
        True
        Check for heterogeneous graphs.
        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
        ... })
        >>> g.is_multigraph
        False
        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1, 1]), torch.tensor([1, 2, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
        ... })
        >>> g.is_multigraph
        True
        """
        return self.gidx.is_multigraph()

    def is_homogeneous(self):
        """Return whether the graph is a homogeneous graph.
        A homogeneous graph only has one node type and one edge type.

        Returns
        -------
        bool
            True if the graph is a homogeneous graph.

        Examples
        --------
        >>> import dgl
        >>> import torch

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))
        >>> g.is_homogeneous
        True

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))})
        >>> g.is_homogeneous
        False
        """
        return len(self.ntypes) == 1 and len(self.etypes) == 1

    def has_nodes(self, vid: jnp.ndarray, ntype: Optional[str]) -> jnp.ndarray:
        """Return whether the graph contains the given nodes.

        Parameters
        ----------
        vid : node ID(s)
            The nodes IDs. The allowed nodes ID formats are:
            * ``int``: The ID of a single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
        ntype : str, optional
            The node type name. Can be omitted if there is
            only one type of nodes in the graph.

        Returns
        -------
        bool or bool Tensor
            A tensor of bool flags where each element is True if the node is in the graph.
            If the input is a single node, return one bool value.

        Examples
        --------
        >>> import dgl
        >>> import torch
        Create a graph with two node types -- 'user' and 'game'.
        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([0, 1]))
        ... })
        Query for the nodes.
        >>> g.has_nodes(0, 'user')
        True
        >>> g.has_nodes(3, 'game')
        False
        >>> g.has_nodes(torch.tensor([3, 0, 1]), 'game')
        tensor([False,  True,  True])
        """
        ntype_idx = self.get_ntype_id(ntype)
        return self.gidx.has_nodes(vid, ntype=ntype_idx)

    def has_edges_between(
            self, u: jnp.ndarray, v: jnp.ndarray, etype: Optional[str]=None,
        ) -> jnp.ndarray:
        """Return whether the graph contains the given edges.

        Parameters
        ----------
        u : node IDs
            The source node IDs of the edges. The allowed formats are:
            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
        v : node IDs
            The destination node IDs of the edges. The allowed formats are:
            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:
            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.
            Can be omitted if the graph has only one type of edges.
        Returns
        -------
        bool or bool Tensor
            A tensor of bool flags where each element is True if the node is in the graph.
            If the input is a single node, return one bool value.
        Examples
        --------
        The following example uses PyTorch backend.
        >>> import dgl
        >>> import torch
        Create a homogeneous graph.
        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))
        Query for the edges.
        >>> g.has_edges_between(1, 2)
        True
        >>> g.has_edges_between(torch.tensor([1, 2]), torch.tensor([2, 3]))
        tensor([ True, False])
        If the graph has multiple edge types, one need to specify the edge type.
        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        ... })
        >>> g.has_edges_between(torch.tensor([1, 2]), torch.tensor([2, 3]), 'plays')
        tensor([ True, False])
        Use a canonical edge type instead when there is ambiguity for an edge type.
        >>> g.has_edges_between(torch.tensor([1, 2]), torch.tensor([2, 3]),
        ...                     ('user', 'follows', 'user'))
        tensor([ True, False])
        >>> g.has_edges_between(torch.tensor([1, 2]), torch.tensor([2, 3]),
        ...                     ('user', 'follows', 'game'))
        tensor([True, True])
        """
        etype_idx = self.get_etype_id(etype)
        return self.gidx.has_edges_between(u, v, etype=etype_idx)

    def find_edges(self, eid: jnp.ndarray, etype=None):
        """Return the source and destination node ID(s) given the edge ID(s).
        Parameters
        ----------
        eid : edge ID(s)
            The edge IDs. The allowed formats are:
            * ``int``: A single ID.
            * Int Tensor: Each element is an ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is an ID.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:
            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.
            Can be omitted if the graph has only one type of edges.
        Returns
        -------
        Tensor
            The source node IDs of the edges. The i-th element is the source node ID of
            the i-th edge.
        Tensor
            The destination node IDs of the edges. The i-th element is the destination node
            ID of the i-th edge.
        Examples
        --------
        The following example uses PyTorch backend.
        >>> import dgl
        >>> import torch
        Create a homogeneous graph.
        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))
        Find edges of IDs 0 and 2.
        >>> g.find_edges(torch.tensor([0, 2]))
        (tensor([0, 1]), tensor([1, 2]))
        For a graph of multiple edge types, it is required to specify the edge type in query.
        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })
        >>> hg.find_edges(torch.tensor([1, 0]), 'plays')
        (tensor([4, 3]), tensor([6, 5]))
        """
        return self.gidx.find_edges(eid=eid, etype=self.get_etype_id(etype))

        
