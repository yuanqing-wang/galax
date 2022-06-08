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

    @property
    def canonical_etypes(self):
        """Return all the canonical edge types in the graph.

        A canonical edge type is a string triplet ``(str, str, str)``
        for source node type, edge type and destination node type.

        Returns
        -------
        list[(str, str, str)]
            All the canonical edge type triplets in a list.

        See Also
        --------
        etypes

        Examples
        --------
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
        src = self.ntypes[self.gidx.metagraph.src]
        dst = self.gidx.metagraph.src

        return list(
            *zip(
                [self.ntypes[idx] for idx in self.gidx.metagraph.src],
                self.etypes,
                [self.ntypes[idx] for idx in self.gidx.metagraph.dst],
            )
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
        if etype is None:
            assert len(self.etypes) == 1, "Etype needs to be specified. "
            etype = self.etypes[0]
        etype_idx = self._etype_invmap[etype]
        ntypes = self.ntypes
        node_frames = self.node_frames
        if len(eids) == len(self.gidx.edges[etype_idx]):
            etypes = self.etypes[:etype_idx] + self.etypes[etype_idx+1:]
            # gidx = self.gidx.
