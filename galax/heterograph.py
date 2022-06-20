"""Module for heterograph definition."""

from typing import (
    Any,
    Mapping,
    Union,
    Optional,
    Tuple,
    Sequence,
    NamedTuple,
    Callable,
)
from collections import namedtuple
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze, unfreeze
from .graph_index import GraphIndex
from .heterograph_index import HeteroGraphIndex
from .view import NodeView, EdgeView, NodeDataView, EdgeDataView
from .function import apply_edges, apply_nodes, ReduceFunction
from .core import message_passing

NodeSpace = namedtuple("NodeSpace", ["data"])
EdgeSpace = namedtuple("EdgeSpace", ["data"])


class HeteroGraph(NamedTuple):
    """Class for storing graph structure and node/edge feature data.

    Parameters
    ----------
    gidx : Optional[HeteroGraphIndex]
        Graph index object.
    ntypes : Optional[Sequence[str]]
        Node type list. ``ntypes[i]`` stores the name of node type i.
        If a pair is given, the graph created is a
        uni-directional bipartite graph,
        and its SRC node types and DST node types are given as in the pair.
    etypes :Optional[Sequence[str]]
        Edge type list. ``etypes[i]`` stores the name of edge type i.
    node_frames : list[Frame]
        Node feature storage. If None, empty frame is created.
        Otherwise, ``node_frames[i]`` stores the node features
        of node type i. (default: None)
    edge_frames : list[Frame], optional
        Edge feature storage. If None, empty frame is created.
        Otherwise, ``edge_frames[i]`` stores the edge features
        of edge type i. (default: None)

    """

    gidx: Optional[HeteroGraphIndex] = None
    node_frames: Optional[NamedTuple] = None
    edge_frames: Optional[NamedTuple] = None
    metamap: Optional[FrozenDict] = None

    @classmethod
    def init(
        cls,
        gidx: Optional[HeteroGraphIndex] = None,
        ntypes: Optional[Sequence[str]] = ("N_",),
        etypes: Optional[Sequence[str]] = ("E_",),
        node_frames: Optional[NamedTuple] = None,
        edge_frames: Optional[NamedTuple] = None,
    ):
        if gidx is None:
            gidx = HeteroGraphIndex()

        if node_frames is None:
            node_frames = [None for _ in range(len(ntypes))]
        if edge_frames is None:
            edge_frames = [None for _ in range(len(etypes))]

        node_frames = namedtuple("node_frames", ntypes)(*node_frames)
        edge_frames = namedtuple("edge_frames", etypes)(*edge_frames)

        # flattened version of metagraph
        src, dst, eid = gidx.metagraph.all_edges()
        src, dst, eid = src.tolist(), dst.tolist(), eid.tolist()
        metamap = {
            _eid: (jnp.zeros(_src), jnp.zeros(_dst))
            for _src, _dst, _eid in zip(src, dst, eid)
        }
        metamap = FrozenDict(metamap)
        return HeteroGraph(
            gidx=gidx,
            node_frames=node_frames,
            edge_frames=edge_frames,
            metamap=metamap,
        )

    def get_meta_edge(self, eid):
        src, dst = self.metamap[eid]
        src = len(src)
        dst = len(dst)
        return src, dst

    @property
    def ntypes(self):
        return self.node_frames._fields

    @property
    def etypes(self):
        return self.edge_frames._fields

    @property
    def _ntype_invmap(self):
        fields = self.node_frames._fields
        return dict(zip(fields, range(len(fields))))

    @property
    def _etype_invmap(self):
        fields = self.edge_frames._fields
        return dict(zip(fields, range(len(fields))))

    def add_nodes(
        self,
        num: int,
        data: Optional[dict] = None,
        ntype: Optional[dict] = None,
    ):
        """Add new nodes of the same node type

        Parameters
        ----------
        num : int
            Number of nodes to add.
        data : Mapping, optional
            Feature data of the added nodes.
        ntype : str, optional
            The type of the new nodes. Can be omitted if there is
            only one node type in the graph.

        Examples
        --------
        **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**
        >>> g = graph(((0, 1), (1, 2)))
        >>> int(g.number_of_nodes())
        3
        >>> g = g.add_nodes(2)
        >>> int(g.number_of_nodes())
        5

        If the graph has some node features and new nodes are added without
        features, their features will be zeros.
        >>> g = g.set_ndata("h", jnp.ones((5, 1)))
        >>> g.ndata["h"].flatten().tolist()
        [1.0, 1.0, 1.0, 1.0, 1.0]

        >>> g = g.add_nodes(1)
        >>> g.ndata["h"].flatten().tolist()
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]

        We can also assign features for the new nodes in adding new nodes.
        >>> g = g.add_nodes(1, {'h': jnp.ones((1, 1)), 'w': jnp.ones((1, 1))})
        >>> g.ndata['h'].flatten().tolist()
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
        >>> g.ndata['w'].flatten().tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        **Heterogeneous Graphs with Multiple Node Types**
        >>> g = graph({
        ...     ('user', 'plays', 'game'): (jnp.array([0, 1, 1, 2]),
        ...                                 jnp.array([0, 0, 1, 1])),
        ...     ('developer', 'develops', 'game'): (jnp.array([0, 1]),
        ...                                         jnp.array([0, 1]))
        ...     })
        >>> g.number_of_nodes("user").item()
        3
        >>> g = g.add_nodes(2, ntype="user")
        >>> g.number_of_nodes("user").item()
        5

        """
        if ntype is None:
            assert len(self.ntypes) == 1, "Please specify node types. "
            ntype = self.ntypes[0]

        if ntype not in self.ntypes:  # new node
            gidx = self.gidx.add_nodes(ntype=len(self.ntypes), num=num)
            ntypes = self.ntypes + (ntype,)
            etypes = self.etypes
            if data is not None:
                data = FrozenDict(data)
                node_frames = self.node_frames + (data,)
            else:
                node_frames = self.node_frames + (None,)
            edge_frames = self.edge_frames

        else:  # existing node
            ntype_idx = self._ntype_invmap[ntype]
            gidx = self.gidx.add_nodes(ntype=ntype_idx, num=num)
            ntypes = self.ntypes
            etypes = self.etypes
            if data is None:
                if self.node_frames[ntype_idx] is None:
                    node_frames = self.node_frames
                else:
                    original_data = self.node_frames[ntype_idx]
                    new_data = FrozenDict(
                        {
                            key: jnp.concatenate(
                                [
                                    original_data[key],
                                    jnp.zeros(
                                        (num,) + original_data[key].shape[1:]
                                    ),
                                ]
                            )
                            for key in original_data.keys()
                        }
                    )

                    node_frames = (
                        self.node_frames[:ntype_idx]
                        + (new_data,)
                        + self.node_frames[ntype_idx + 1 :]
                    )

            else:
                new_data = {}
                original_data = self.node_frames[ntype_idx]

                for key in original_data:
                    if key in data:
                        value = jnp.concatenate(
                            [
                                original_data[key],
                                data[key],
                            ]
                        )
                    else:
                        placeholder = jnp.zeros(
                            (num,) + original_data[key].shape[1:]
                        )

                        value = jnp.concatenate(
                            [
                                original_data[key],
                                placeholder,
                            ]
                        )

                    new_data[key] = value

                for key in data:
                    if key not in original_data:
                        placeholder = jnp.zeros(
                            (self.number_of_nodes(ntype),)
                            + data[key].shape[1:]
                        )
                        value = jnp.concatenate(
                            [
                                placeholder,
                                data[key],
                            ]
                        )

                        new_data[key] = value

                new_data = FrozenDict(new_data)

                node_frames = (
                    self.node_frames[:ntype_idx]
                    + (new_data,)
                    + self.node_frames[ntype_idx + 1 :]
                )

            edge_frames = self.edge_frames

        return self.__class__.init(
            gidx=gidx,
            ntypes=ntypes,
            etypes=etypes,
            node_frames=node_frames,
            edge_frames=edge_frames,
        )

    def add_edges(
        self,
        u: jnp.ndarray,
        v: jnp.ndarray,
        data: Optional[jnp.ndarray] = None,
        etype: Optional[str] = None,
        srctype: Optional[str] = None,
        dsttype: Optional[str] = None,
    ):
        """Add multiple new edges for the specified edge type
        The i-th new edge will be from ``u[i]`` to ``v[i]``.

        Parameters
        ----------
        u : jnp.ndarray
            Source node IDs,
            ``u[i]`` gives the source node for the i-th new edge.
        v : jnp.ndarray
            Destination node IDs, `
            `v[i]`` gives the destination node for the i-th new edge.
        data : dict, optional
            Feature data of the added edges.
            The i-th row of the feature data
            corresponds to the i-th new edge.
        etype : str or tuple of str, optional
            The type of the new edges. Can be omitted if there is
            only one edge type in the graph.

        Notes
        -----
        * If the key of ``data`` does not contain some existing feature
          fields, those features for the new edges will be filled
          with zeros.
        * If the key of ``data`` contains new feature fields,
          those features for the old edges will be filled with zeros.

        Examples
        --------
        >>> g = graph(((0, 1), (1, 2)))
        >>> g.number_of_edges()
        2

        >>> g = g.add_nodes(2)
        >>> g = g.add_edges((1, 3), (0, 1))
        >>> g.number_of_edges()
        4

        If the graph has some edge features and new edges are added without
        features, their features will be create.
        >>> g = g.set_edata("h", jnp.ones((4, 1)))
        >>> g = g.add_edges((1, ), (1, ))
        >>> g.edata['h'].flatten().tolist()
        [1.0, 1.0, 1.0, 1.0, 0.0]

        We can also assign features for the new edges in adding new edges.
        >>> g = g.add_edges(jnp.array([0, 0]), jnp.array([2, 2]),
        ...     {'h': jnp.array([[1.], [2.]]), 'w': jnp.ones((2, 1))})
        >>> g.edata['h'].flatten().tolist()
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0]

        Since ``data`` contains new feature fields, the features for old edges
        will be created.
        >>> g.edata['w'].flatten().tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

        **Heterogeneous Graphs with Multiple Edge Types**
        >>> g = graph({
        ...     ('user', 'plays', 'game'): ((0, 1, 1, 2),
        ...                                 (0, 0, 1, 1)),
        ...     ('developer', 'develops', 'game'): ((0, 1),
        ...                                         (0, 1))
        ...     })
        >>> g.number_of_edges('plays')
        4
        >>> g = g.add_edges(jnp.array([2]), jnp.array([1]), etype='plays')
        >>> g.number_of_edges('plays')
        5

        """
        if etype is None:
            assert len(self.etypes) == 1, "Etype needs to be specified. "
            etype = self.etypes[0]

        # convert nodes to ndarray
        if not isinstance(u, jnp.ndarray):
            u = jnp.array(u)
        if not isinstance(v, jnp.ndarray):
            v = jnp.array(v)

        if etype not in self.etypes:  # new node
            etype_idx = len(self.etypes)
            gidx = self.gidx.add_edges(
                etype=etype_idx,
                src=u,
                dst=v,
                srctype=srctype,
                dsttype=dsttype,
            )
            ntypes = self.ntypes
            etypes = self.etypes + (etype,)
            if data is not None:
                data = FrozenDict(data)
                edge_frames = self.edge_frames + (data,)
            else:
                edge_frames = self.edge_frames + (None,)
            node_frames = self.node_frames

        else:  # existing node
            etype_idx = self._etype_invmap[etype]
            gidx = self.gidx.add_edges(etype=etype_idx, src=u, dst=v)
            ntypes = self.ntypes
            etypes = self.etypes
            if data is None:
                if self.edge_frames[etype_idx] is None:
                    edge_frames = self.edge_frames
                else:
                    original_data = self.edge_frames[etype_idx]
                    new_data = FrozenDict(
                        {
                            key: jnp.concatenate(
                                [
                                    original_data[key],
                                    jnp.zeros(
                                        (len(u),)
                                        + original_data[key].shape[1:]
                                    ),
                                ]
                            )
                            for key in original_data.keys()
                        }
                    )

                    edge_frames = (
                        self.edge_frames[:etype_idx]
                        + (new_data,)
                        + self.edge_frames[etype_idx + 1 :]
                    )

            else:
                new_data = {}
                original_data = self.edge_frames[etype_idx]

                for key in original_data:
                    if key in data:
                        value = jnp.concatenate(
                            [
                                original_data[key],
                                data[key],
                            ]
                        )
                    else:
                        placeholder = jnp.zeros(
                            (num,) + original_data[key].shape[1:]
                        )

                        value = jnp.concatenate(
                            [
                                original_data[key],
                                placeholder,
                            ]
                        )

                    new_data[key] = value

                for key in data:
                    if key not in original_data:
                        placeholder = jnp.zeros(
                            (self.number_of_edges(etype),)
                            + data[key].shape[1:]
                        )
                        value = jnp.concatenate(
                            [
                                placeholder,
                                data[key],
                            ]
                        )

                        new_data[key] = value

                new_data = FrozenDict(new_data)

                edge_frames = (
                    self.edge_frames[:etype_idx]
                    + (new_data,)
                    + self.edge_frames[etype_idx + 1 :]
                )

            node_frames = self.node_frames

        return self.__class__.init(
            gidx=gidx,
            ntypes=ntypes,
            etypes=etypes,
            node_frames=node_frames,
            edge_frames=edge_frames,
        )

    def remove_edges(
        self, eids: Optional[jnp.array] = None, etype: Optional[str] = None
    ):
        """Remove multiple edges with the specified edge type
        Nodes will not be removed. After removing edges, the rest
        edges will be re-indexed using consecutive integers from 0,
        with their relative order preserved.
        The features for the removed edges will be removed accordingly.

        Parameters
        ----------
        eids : int, tensor, numpy.ndarray, list
            IDs for the edges to remove.
        etype : str or tuple of str, optional
            The type of the edges to remove. Can be omitted if there is
            only one edge type in the graph.

        Examples
        --------
        >>> g = graph(((0, 0, 2), (0, 1, 2)))
        >>> g = g.set_edata("he", jnp.array([0.0, 1.0, 2.0]))
        >>> g = g.remove_edges((0, 1))
        >>> int(g.number_of_edges())
        1

        **Heterogeneous Graphs with Multiple Edge Types**
        >>> g = graph({
        ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        ...     ('developer', 'develops', 'game'): ([0, 1], [0, 1]),
        ...     })

        >>> g = g.remove_edges([0, 1], 'plays')
        >>> int(g.number_of_edges("plays"))
        2


        """

        etype_idx = self.get_etype_id(etype)  # get the id of the edge type

        if not isinstance(eids, jnp.ndarray):  # make eid into array
            eids = jnp.array(eids)

        assert len(eids) <= self.number_of_edges(etype), "Not enough edges. "
        if len(eids) == self.number_of_edges(etype):
            complete = True
        else:
            complete = False

        if complete:  # delete etype entirely
            etypes = self.etypes[:etype_idx] + self.etypes[etype_idx + 1 :]
            edge_frames = (
                self.edge_frames[:etype_idx]
                + self.edge_frames[etype_idx + 1 :]
            )

        else:  # partially delete
            etypes = self.etypes
            edge_frames = self.edge_frames
            sub_edge_frame = edge_frames[etype_idx]
            if sub_edge_frame is not None:
                sub_edge_frame = FrozenDict(
                    {
                        key: jnp.delete(value, eids)
                        for key, value in sub_edge_frame.items()
                    }
                )

                edge_frames = (
                    self.edge_frames[:etype_idx]
                    + (sub_edge_frame,)
                    + self.edge_frames[etype_idx + 1 :]
                )

        gidx = self.gidx.remove_edges(etype=etype_idx, eids=eids)
        return self.__class__.init(
            gidx=gidx,
            ntypes=self.ntypes,
            etypes=etypes,
            node_frames=self.node_frames,
            edge_frames=edge_frames,
        )

    def remove_nodes(self, nids: jnp.ndarray, ntype: Optional[str] = None):
        """Remove multiple nodes with the specified node type
        Edges that connect to the nodes will be removed as well. After removing
        nodes and edges, the rest nodes and edges will be re-indexed using
        consecutive integers from 0, with their relative order preserved.
        The features for the removed nodes/edges will be removed accordingly.

        Parameters
        ----------
        nids : int, tensor, numpy.ndarray, list
            Nodes to remove.
        ntype : str, optional
            The type of the nodes to remove. Can be omitted if there is
            only one node type in the graph.

        Notes
        -----
        * This does not remove the etype entirely.

        Examples
        --------
        **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**
        >>> g = graph(([0, 0, 2], [0, 1, 2]))
        >>> g = g.set_ndata("hv", jnp.array([0.0, 1.0, 2.0]))
        >>> g = g.set_edata("he", jnp.array([0.0, 1.0, 2.0]))
        >>> g = g.remove_nodes((0, 1))
        >>> int(g.number_of_nodes())
        1

        >>> g.ndata["hv"].flatten().tolist()
        [2.0]
        >>> g.edata["he"].flatten().tolist()
        [2.0]

        **Heterogeneous Graphs with Multiple Node Types**
        >>> g = graph({
        ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        ...     ('developer', 'develops', 'game'): ([0, 1], [0, 1])
        ...     })
        >>> g = g.remove_nodes([0, 1], ntype='game')
        >>> g.number_of_nodes('user').item()
        3
        >>> 'game' in g.ntypes
        False

        """
        ntype_idx = self.get_ntype_id(ntype)  # get ntype

        if not isinstance(nids, jnp.ndarray):  # make nid into array
            nids = jnp.array(nids)

        assert len(nids) <= self.number_of_nodes(ntype), "Not enough edges. "
        if len(nids) == self.number_of_nodes(ntype):
            complete = True
        else:
            complete = False

        if complete:  # delete etype entirely
            ntypes = self.ntypes[:ntype_idx] + self.ntypes[ntype_idx + 1 :]
            node_frames = (
                self.node_frames[:ntype_idx]
                + self.node_frames[ntype_idx + 1 :]
            )

        else:  # partially delete
            ntypes = self.ntypes
            node_frames = self.node_frames
            sub_node_frame = node_frames[ntype_idx]
            if sub_node_frame is not None:
                sub_node_frame = FrozenDict(
                    {
                        key: jnp.delete(value, nids)
                        for key, value in sub_node_frame.items()
                    }
                )

                node_frames = (
                    self.node_frames[:ntype_idx]
                    + (sub_node_frame,)
                    + self.node_frames[ntype_idx + 1 :]
                )

        gidx = self.gidx.remove_nodes(ntype=ntype_idx, nids=nids)

        # take care of edge_frames
        _, __, in_edge_types = self.gidx.metagraph.in_edges(ntype_idx)
        _, __, out_edge_types = self.gidx.metagraph.out_edges(ntype_idx)

        edge_frames = list(self.edge_frames)
        for edge_type in range(self.gidx.metagraph.number_of_edges()):
            if edge_frames[edge_type] is None:
                continue
            else:
                if edge_type in in_edge_types and edge_type in out_edge_types:
                    if self.gidx.edges[edge_type] is None:
                        continue
                    else:
                        src, dst = self.gidx.edges[edge_type]
                        v_is_src = jnp.expand_dims(
                            src, -1
                        ) == jnp.expand_dims(nids, 0)
                        v_is_dst = jnp.expand_dims(
                            dst, -1
                        ) == jnp.expand_dims(nids, 0)
                        v_in_edge = (v_is_src + v_is_dst).any(-1)
                        edge_frames[edge_type] = FrozenDict(
                            {
                                key: value[~v_in_edge]
                                for key, value in edge_frames[
                                    edge_type
                                ].items()
                            }
                        )
                elif edge_type in in_edge_types:
                    if self.gidx.edges[edge_type] is None:
                        continue
                    else:
                        src, dst = self.gidx.edges[edge_type]
                        v_is_dst = jnp.expand_dims(
                            dst, -1
                        ) == jnp.expand_dims(nids, 0)
                        v_in_edge = v_is_dst
                        edge_frames[edge_type] = FrozenDict(
                            {
                                key: value[~v_in_edge]
                                for key, value in edge_frames[
                                    edge_type
                                ].items()
                            }
                        )

                elif edge_type in out_edge_types:
                    if self.gidx.edges[edge_type] is None:
                        continue
                    else:
                        src, dst = self.gidx.edges[edge_type]
                        v_is_src = jnp.expand_dims(
                            src, -1
                        ) == jnp.expand_dims(nids, 0)
                        v_in_edge = v_is_src
                        edge_frames[edge_type] = FrozenDict(
                            {
                                key: value[~v_in_edge]
                                for key, value in edge_frames[
                                    edge_type
                                ].items()
                            }
                        )

        edge_frames = tuple(edge_frames)

        return self.__class__.init(
            gidx=gidx,
            ntypes=ntypes,
            etypes=self.etypes,
            node_frames=node_frames,
            edge_frames=edge_frames,
        )

    @property
    def nodes(self):
        return NodeView(self)

    @property
    def edges(self):
        return EdgeView(self)

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
        """Convert an edge type to the corresponding canonical
        edge type in the graph.

        A canonical edge type is a string triplet ``(str, str, str)``
        for source node type, edge type and destination node type.
        The function expects the given edge type name can uniquely identify
        a canonical edge type.

        Parameters
        ----------
        etype : str
            If :attr:`etype` is an edge type (str),
            it returns the corresponding canonical edge type in the graph.

        Returns
        -------
        (str, str, str)
            The canonical edge type corresponding to the edge type.

        """
        etype_idx = self.get_etype_id(etype)
        src, dst = self.gidx.metagraph.find_edge(etype_idx)
        return self.ntypes[src], etype, self.ntyes[dst]

    def get_ntype_id(self, ntype: Optional[str] = None) -> int:
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
            # assert len(self.ntypes) == 1, "Ntype needs to be specified. "
            return 0
        else:
            assert ntype in self._ntype_invmap, "No such ntype %s. " % ntype
            return self._ntype_invmap[ntype]

    def get_etype_id(self, etype: Optional[str] = None) -> int:
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
            # assert len(self.etypes) == 1, "Etype needs to be specified. "
            return 0
        else:
            assert etype in self._etype_invmap, "No such etype %s. " % etype
            return self._etype_invmap[etype]

    def number_of_nodes(self, ntype: Optional[str] = None):
        """Return the number of nodes with ntype.

        Parameters
        ----------
        ntype : str
            Node type.

        Return
        ------
        int
            Number of nodes.

        Examples
        --------

        """
        ntype_idx = self.get_ntype_id(ntype)
        if self.node_frames[ntype_idx] is None:
            return self.gidx.number_of_nodes(self.get_ntype_id(ntype))
        elif len(self.node_frames[ntype_idx]) == 0:
            return self.gidx.number_of_nodes(self.get_ntype_id(ntype))
        else:
            return len(next(iter(self.node_frames[ntype_idx].values())))

    def number_of_edges(self, etype: Optional[str] = None):
        """Return the number of nodes with ntype.

        Parameters
        ----------
        etype : str
            Edge type.

        Return
        ------
        int
            Number of edges.
        """
        etype_idx = self.get_etype_id(etype)
        if self.edge_frames[etype_idx] is None:
            return self.gidx.number_of_edges(self.get_etype_id(etype))
        elif len(self.edge_frames[etype_idx]) == 0:
            return self.gidx.number_of_edges(self.get_etype_id(etype))
        else:
            return len(next(iter(self.edge_frames[etype_idx].values())))

    def is_multigraph(self):
        """Return whether the graph is a multigraph with parallel edges.
        A multigraph has more than one edges between the same pair of nodes,
        called *parallel edges*.
        For heterogeneous graphs, parallel edge further requires
        the canonical edge type to be the same
        (see :meth:`canonical_etypes` for the
        definition).

        Returns
        -------
        bool
            True if the graph is a multigraph.

        """
        return self.gidx.is_multigraph()

    def is_homogeneous(self):
        """Return whether the graph is a homogeneous graph.
        A homogeneous graph only has one node type and one edge type.

        Returns
        -------
        bool
            True if the graph is a homogeneous graph.

        """
        return len(self.ntypes) == 1 and len(self.etypes) == 1

    def has_nodes(
        self, vid: jnp.ndarray, ntype: Optional[str]
    ) -> jnp.ndarray:
        """Return whether the graph contains the given nodes.

        Parameters
        ----------
        vid : node ID(s)
            The nodes IDs. The allowed nodes ID formats are:
            * ``int``: The ID of a single node.
            * Int Tensor: Each element is a node ID.
            * iterable[int]: Each element is a node ID.
        ntype : str, optional
            The node type name. Can be omitted if there is
            only one type of nodes in the graph.

        Returns
        -------
        bool or bool Tensor
            A tensor of bool flags where each element is
            True if the node is in the graph.
            If the input is a single node, return one bool value.

        """
        ntype_idx = self.get_ntype_id(ntype)
        return self.gidx.has_nodes(vid, ntype=ntype_idx)

    def has_edges_between(
        self,
        u: jnp.ndarray,
        v: jnp.ndarray,
        etype: Optional[str] = None,
    ) -> jnp.ndarray:
        """Return whether the graph contains the given edges.

        Parameters
        ----------
        u : node IDs
            The source node IDs of the edges. The allowed formats are:
            * ``int``: A single node.
            * Int Tensor: Each element is a node ID.
            * iterable[int]: Each element is a node ID.
        v : node IDs
            The destination node IDs of the edges. The allowed formats are:
            * ``int``: A single node.
            * Int Tensor: Each element is a node ID.
            * iterable[int]: Each element is a node ID.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:
            * ``(str, str, str)`` for source node type,
              edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.
            Can be omitted if the graph has only one type of edges.
        Returns
        -------
        bool or bool Tensor
            A tensor of bool flags where each element is True
            if the node is in the graph.
            If the input is a single node, return one bool value.

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
            * Int Tensor: Each element is an ID.
              The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is an ID.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:
            * ``(str, str, str)`` for source node type,
              edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.
            Can be omitted if the graph has only one type of edges.
        Returns
        -------
        Tensor
            The source node IDs of the edges.
            The i-th element is the source node ID of the i-th edge.
        Tensor
            The destination node IDs of the edges.
            The i-th element is the destination node ID of the i-th edge.

        """
        return self.gidx.find_edges(eid=eid, etype=self.get_etype_id(etype))

    def in_degrees(
        self,
        v: Optional[jnp.ndarray] = None,
        etype: Optional[str] = None,
    ):
        """Return the in-degree(s) of the given nodes.
        It computes the in-degree(s) w.r.t. to the edges of the given edge type.

        Parameters
        ----------
        v : node IDs
            The node IDs. The allowed formats are:
            * ``int``: A single node.
            * Int Tensor: Each element is a node ID.
              The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
            If not given, return the in-degrees of all the nodes.
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:
            * ``(str, str, str)`` for source node type,
              edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.
            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        int or Tensor
            The in-degree(s) of the node(s) in a Tensor.
            The i-th element is the in-degree
            of the i-th input node. If :attr:`v` is an ``int``,
            return an ``int`` too.

        """
        etype_idx = self.get_etype_id(etype)

        if v is None:
            v = jnp.arange(
                self.gidx.n_nodes[self.gidx.metagraph.dst[etype_idx]]
            )

        return self.gidx.in_degrees(
            v=v,
            etype=etype_idx,
        )

    def out_degrees(
            self,
            u: Optional[jnp.ndarray] = None,
            etype: Optional[str] = None
    ):
        """Return the out-degree(s) of the given nodes.
        It computes the out-degree(s) w.r.t. to the edges of the given edge type.

        Parameters
        ----------
        u : node IDs
            The node IDs. The allowed formats are:
            * ``int``: A single node.
            * Int Tensor: Each element is a node ID.
              The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
            If not given, return the in-degrees of all the nodes.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:
            * ``(str, str, str)`` for source node type, edge type
              and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.
            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        int or Tensor
            The out-degree(s) of the node(s) in a Tensor.
            The i-th element is the out-degree
            of the i-th input node. If :attr:`v` is an ``int``,
            return an ``int`` too.


        """
        etype_idx = self.get_etype_id(etype)

        if u is None:
            u = jnp.arange(self.number_of_nodes(etype))

        return self.gidx.out_degrees(
            u=u,
            etype=etype_idx,
        )

    def adjacency_matrix(
        self,
        transpose: bool = False,
        etype: Optional[str] = None,
    ):
        """Return the adjacency matrix of edges of the given edge type.
        By default, a row of returned adjacency matrix represents the
        source of an edge and the column represents the destination.
        When transpose is True, a row represents the destination and a column
        represents the source.
        Parameters
        ----------
        transpose : bool, optional
            A flag to transpose the returned adjacency matrix. (Default: False)
        ctx : context, optional
            The context of returned adjacency matrix. (Default: cpu)
        scipy_fmt : str, optional
            If specified, return a scipy sparse matrix in the given format.
            Otherwise, return a backend dependent sparse tensor. (Default: None)
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:
            * ``(str, str, str)`` for source node type,
              edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.
            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        SparseTensor or scipy.sparse.spmatrix
            Adjacency matrix.

        """
        return self.gidx.adjacency_matrix(
            etype=self.get_etype_id(etype),
            transpose=transpose,
        )

    adj = adjacency_matrix

    def incidence_matrix(self, typestr: str, etype: Optional[str] = None):
        """Return the incidence matrix representation of edges with the given
        edge type.
        An incidence matrix is an n-by-m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.
        There are three types of incidence matrices :math:`I`:
        * ``in``:
            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`
              (or :math:`v` is the dst node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.
        * ``out``:
            - :math:`I[v, e] = 1` if :math:`e` is the out-edge of :math:`v`
              (or :math:`v` is the src node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.
        * ``both`` (only if source and destination node type are the same):
            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`;
            - :math:`I[v, e] = -1` if :math:`e` is the out-edge of :math:`v`;
            - :math:`I[v, e] = 0` otherwise (including self-loop).
        Parameters
        ----------
        typestr : str
            Can be either ``in``, ``out`` or ``both``
        ctx : context, optional
            The context of returned incidence matrix. (Default: cpu)
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:
            * ``(str, str, str)`` for source node type,
              edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.
            Can be omitted if the graph has only one type of edges.
        Returns
        -------
        Framework SparseTensor
            The incidence matrix.

        """
        return self.gidx.incidence_matrix(
            typestr=typestr,
            etype=self.get_etype_id(etype),
        )

    inc = incidence_matrix

    def set_ndata(self, key, data, ntype=None):
        """Set node data.

        Parameters
        ----------
        key : str
            Name of the data field.
        data : jnp.array
            Node data.
        ntype : str
            Node type.

        Returns
        -------
        HeteroGraph
            A new graph with ndata.

        Examples
        --------
        >>> g = graph(((0, 1), (1, 2)))
        >>> g = g.set_ndata('h', jnp.zeros(3))
        """
        if ntype is None:
            ntype = self.ntypes[0]
        node_frame = getattr(self.node_frames, ntype)
        if node_frame is None:
            node_frame = {}
        node_frame = unfreeze(node_frame)
        node_frame[key] = data
        node_frame = freeze(node_frame)
        node_frames = self.node_frames._replace(**{ntype: node_frame})
        return self._replace(node_frames=node_frames)

    def set_edata(self, key, data, etype=None):
        """Set edge data.

        Parameters
        ----------
        key : str
            Name of the data field.
        data : jnp.array
            Node data.
        etype : str
            Edge type.

        Returns
        -------
        HeteroGraph
            A new graph with ndata.

        Examples
        --------
        >>> g = graph(((0, 1), (1, 2)))
        >>> g = g.set_edata('h', jnp.zeros(3))
        """
        if etype is None:
            etype = self.etypes[0]
        edge_frame = getattr(self.edge_frames, etype)
        if edge_frame is None:
            edge_frame = {}
        edge_frame = unfreeze(edge_frame)
        edge_frame[key] = data
        edge_frame = freeze(edge_frame)
        edge_frames = self.edge_frames._replace(**{etype: edge_frame})
        return self._replace(edge_frames=edge_frames)

    @property
    def edata(self):
        """Edge data."""
        return EdgeDataView(self, 0)

    @property
    def ndata(self):
        """Node data."""
        return NodeDataView(self, 0)

    @property
    def srcdata(self, etype: Optional[str] = None):
        """Return a node data view for setting/getting source node features.
        Let ``g`` be a Graph.

        Parameters
        ----------
        etype : Optional[str]
            Edge type.

        Examples
        --------
        >>> g = graph({
        ...     ('user', 'plays', 'game'): ([0, 1], [1, 2])})
        >>> g = g.set_ndata("h", jnp.ones(2), "user")
        >>> g.srcdata["h"].flatten().tolist()
        [1.0, 1.0]

        """
        etype_idx = self.get_etype_id(etype)
        srctype_idx, _ = self.gidx.metagraph.find_edge(etype_idx)
        src, _ = self.gidx.edges[etype_idx]
        node_frame = self.node_frames[srctype_idx]
        _node_frame = FrozenDict(
            {key: value[src] for key, value in node_frame.items()}
        )

        return _node_frame

    @property
    def dstdata(self):
        """Return a node data view for setting/getting destination node features.
        Let ``g`` be a Graph.

        Parameters
        ----------
        etype : Optional[str]
            Edge type.

        Examples
        --------
        >>> g = graph({
        ...     ('user', 'plays', 'game'): ([0, 1], [1, 2])})
        >>> g = g.set_ndata("h", jnp.ones(3), "game")
        >>> g.dstdata["h"].flatten().tolist()
        [1.0, 1.0]

        """

        etype_idx = 0
        _, dsttype_idx = self.gidx.metagraph.find_edge(etype_idx)
        _, dst = self.gidx.edges[etype_idx]
        node_frame = self.node_frames[dsttype_idx]
        _node_frame = FrozenDict(
            {key: value[dst] for key, value in node_frame.items()}
        )

        return _node_frame

    def add_self_loop(self, etype: Optional[str] = None):
        """Add self loop given etype.

        Parameters
        ----------
        etype : Optional[str] = None
            Edge type.

        Returns
        -------
        HeteroGraph
            The resulting graph.

        Examples
        --------
        >>> g = graph(((0, 1), (1, 2)))
        >>> g = g.add_self_loop()
        >>> g.number_of_edges()
        5
        >>> src, dst = g.edges()
        >>> src.tolist(), dst.tolist()
        ([0, 1, 0, 1, 2], [1, 2, 0, 1, 2])

        """
        etype_idx = self.get_etype_id(etype)
        srctype_idx, dsttype_idx = self.gidx.metagraph.find_edge(etype_idx)
        assert srctype_idx == dsttype_idx
        n_nodes = self.gidx.n_nodes[srctype_idx]
        return self.add_edges(
            jnp.arange(n_nodes), jnp.arange(n_nodes), etype=etype,
        )


    def remove_self_loop(self, etype: Optional[str] = None):
        """Add self loop given etype.

        Parameters
        ----------
        etype : Optional[str] = None
            Edge type.

        Returns
        -------
        HeteroGraph
            The resulting graph.

        Examples
        --------
        >>> g = graph(((0, 1, 0, 1, 2), (1, 2, 0, 1, 2)))
        >>> g = g.remove_self_loop()
        >>> g.number_of_edges()
        2
        >>> src, dst = g.edges()
        >>> src.tolist(), dst.tolist()
        ([0, 1], [1, 2])

        """
        etype_idx = self.get_etype_id(etype)
        srctype_idx, dsttype_idx = self.gidx.metagraph.find_edge(etype_idx)
        assert srctype_idx == dsttype_idx
        src, dst = self.gidx.edges[etype_idx]
        eid = jnp.where(src == dst)[0]
        return self.remove_edges(eid, etype=etype)

    @classmethod
    def from_dgl(cls, graph):
        """Construct a heterograph from dgl.DGLGraph

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.

        Returns
        -------
        HeteroGraph
            The resulting graph.

        """
        # resursively construct the heterograph index object
        heterograph_index = HeteroGraphIndex.from_dgl(graph._graph)
        ntypes = graph.ntypes
        etypes = graph.etypes

        # replace node and edge type so that it doesn't error for namedtuple
        ntypes = [ntype.replace("_N", "N_") for ntype in ntypes]
        etypes = [etype.replace("_E", "E_") for etype in etypes]

        to_jnp = lambda frame: {
            key: jnp.array(value)
            if not isinstance(value, jnp.ndarray) else value
            for key, value in frame.items()
        }

        # copy frames
        node_frames = [to_jnp(dict(frame)) for frame in graph._node_frames]
        edge_frames = [to_jnp(dict(frame)) for frame in graph._edge_frames]

        return cls.init(
            gidx=heterograph_index,
            ntypes=ntypes,
            etypes=etypes,
            node_frames=node_frames,
            edge_frames=edge_frames,
        )

    def apply_nodes(
            self,
            apply_function: Callable,
            in_field: Optional[str] = "h",
            out_field: Optional[str] = None,
            ntype: Optional[str] = None,
    ):
        """Alias to function.apply_nodes."""

        apply_function = apply_nodes(
            apply_function,
            in_field=in_field,
            out_field=out_field,
            ntype=ntype,
        )
        return apply_function(self)

    def apply_edges(
            self,
            apply_function: Callable,
            in_field: Optional[str] = "h",
            out_field: Optional[str] = None,
            etype: Optional[str] = None,
    ):
        """Alias to function.apply_edges."""
        apply_function = apply_edges(
            apply_function,
            in_field=in_field,
            out_field=out_field,
            etype=etype,
        )
        return apply_function(self)

    def update_all(
        self,
        mfunc: Optional[Callable],
        rfunc: Optional[ReduceFunction],
        afunc: Optional[Callable] = None,
        etype: Optional[Callable] = None,
    ):
        """Alias to core.message_passing.

        Parameters
        ----------
        mfunc : Callable
            Message function.
        rfunc : Callable
            Reduce function.
        afunc : Callable
            Apply function.

        Returns
        -------
        HeteroGraph
            The resulting graph.

        Examples
        --------
        >>> import galax
        >>> import jax
        >>> import jax.numpy as jnp
        >>> g = galax.graph(((0, 1), (1, 2)))
        >>> g = g.ndata.set("h", jnp.ones(3))
        >>> mfunc = galax.function.copy_u("h", "m")
        >>> rfunc = galax.function.sum("m", "h1")
        >>> _g = g.update_all(mfunc, rfunc)
        >>> _g.ndata['h1'].flatten().tolist()
        [0.0, 1.0, 1.0]

        """
        return message_passing(
            graph=self, mfunc=mfunc, rfunc=rfunc, afunc=afunc, etype=etype,
        )

def graph(
    data: Any,
    n_nodes: Optional[Union[Mapping, int]] = None,
):
    """Create a heterogeneous graph and return.

    Parameters
    ----------
    data : Any
    n_nodes : Optional[Union[Mapping, int]] (default=None)

    Returns
    -------
    HeteroGraph
        The created graph.

    Examples
    --------
    Create a small three-edge graph.
    >>> # Source nodes for edges (2, 1), (3, 2), (4, 3)
    >>> src_ids = jnp.array([2, 3, 4])
    >>> # Destination nodes for edges (2, 1), (3, 2), (4, 3)
    >>> dst_ids = jnp.array([1, 2, 3])
    >>> g = graph((src_ids, dst_ids))
    >>> int(g.number_of_nodes())
    5
    >>> int(g.number_of_edges())
    3
    >>> g.ntypes
    ('N_',)
    >>> g.etypes
    ('E_',)

    Explicitly specify the number of nodes in the graph.
    >>> g = graph((src_ids, dst_ids), n_nodes=2666)
    >>> int(g.number_of_nodes())
    2666
    >>> int(g.number_of_edges())
    3

    >>> data_dict = {
    ...     ('user', 'follows', 'user'): ((0, 1), (1, 2)),
    ...     ('user', 'follows_', 'topic'): ((1, 1), (1, 2)), # etype different
    ...     ('user', 'plays', 'game'): ((0, 3), (3, 4)),
    ... }
    >>> g = graph(data_dict)
    >>> g.number_of_nodes('user').item()
    4
    >>> int(g.number_of_edges('follows'))
    2

    """
    if isinstance(data, tuple):  # single node type, single edge type
        metagraph = GraphIndex(
            n_nodes=1, src=jnp.array([0]), dst=jnp.array([0])
        )

        assert len(data) == 2, "Only need src and dst. "
        src, dst = data
        if not isinstance(src, jnp.ndarray):
            src = jnp.array(src)
        if not isinstance(dst, jnp.ndarray):
            dst = jnp.array(dst)

        edges = ((src, dst),)
        inferred_n_nodes = max(max(edges[0][0]), max(edges[0][1])).item() + 1
        if n_nodes is None:  # infer n_nodes
            n_nodes = inferred_n_nodes
        else:
            assert isinstance(n_nodes, int), "Single node type."
            assert (
                n_nodes >= inferred_n_nodes
            ), "Edge with non-existing nodes. "
        n_nodes = jnp.array([n_nodes])
        gidx = HeteroGraphIndex(
            metagraph=metagraph,
            n_nodes=n_nodes,
            edges=edges,
        )
        return HeteroGraph.init(gidx=gidx)

    elif isinstance(data, Mapping):
        metagraph = GraphIndex()
        from collections import OrderedDict

        _ntype_invmap = OrderedDict()
        _etype_invmap = OrderedDict()

        edges = []
        inferred_n_nodes = []
        for key, value in data.items():
            assert isinstance(key, tuple), "Edge has to be tuple. "
            assert isinstance(value, tuple), "Edge has to be tuple. "
            assert len(key) == 3, "Specify src, etype, and dst. "
            srctype, etype, dsttype = key

            # put ntype into invmap
            if srctype not in _ntype_invmap:
                _ntype_invmap[srctype] = len(_ntype_invmap)
                metagraph = metagraph.add_nodes(1)
                inferred_n_nodes.append(0)
            srctype_idx = _ntype_invmap[srctype]

            if dsttype not in _ntype_invmap:
                _ntype_invmap[dsttype] = len(_ntype_invmap)
                metagraph = metagraph.add_nodes(1)
                inferred_n_nodes.append(0)
            dsttype_idx = _ntype_invmap[dsttype]

            # put etype into invmap
            assert etype not in _etype_invmap, "Etype has to be unique. "
            _etype_invmap[etype] = len(_etype_invmap)
            metagraph = metagraph.add_edge(srctype_idx, dsttype_idx)

            assert len(value) == 2, "Only need src and dst. "
            src, dst = value
            if not isinstance(src, jnp.ndarray):
                src = jnp.array(src)
            if not isinstance(dst, jnp.ndarray):
                dst = jnp.array(dst)
            edges.append((src, dst))

            inferred_n_nodes[srctype_idx] = max(
                inferred_n_nodes[srctype_idx], max(src).item() + 1
            )

            inferred_n_nodes[dsttype_idx] = max(
                inferred_n_nodes[dsttype_idx], max(dst).item() + 1
            )

        inferred_n_nodes = jnp.array(inferred_n_nodes)

        # edges and n_nodes
        edges = tuple(edges)

        for key, value in data.items():
            srctype, etype, dsttype = key
            src, dst = value

        # custom n_nodes
        if n_nodes is not None:
            assert (n_nodes >= inferred_n_nodes).all()
            n_nodes = jnp.max(n_nodes, inferred_n_nodes)
        else:
            n_nodes = inferred_n_nodes

        # organize gidx
        gidx = HeteroGraphIndex(
            metagraph=metagraph,
            n_nodes=n_nodes,
            edges=edges,
        )

        # extract ntypes and etypes from ordered dict
        ntypes = tuple(_ntype_invmap.keys())
        etypes = tuple(_etype_invmap.keys())

        return HeteroGraph.init(
            gidx=gidx,
            ntypes=ntypes,
            etypes=etypes,
        )

def from_dgl(graph):
    """Construct a heterograph from dgl.DGLGraph

    Parameters
    ----------
    graph : dgl.DGLGraph
        Input graph.

    Returns
    -------
    HeteroGraph
        The resulting graph.

    """
    return HeteroGraph.from_dgl(graph)
