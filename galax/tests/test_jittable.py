import jax
import jax.numpy as jnp

def test_graph_index_pytree():
    from galax.graph_index import GraphIndex
    g = GraphIndex(
        2,
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array([0, 2], dtype=jnp.int32)
    )
    children, aux_data = g.tree_flatten()
    _g = g.tree_unflatten(aux_data, children)
    assert _g.n_nodes == g.n_nodes
    assert (_g.src == g.src).all()
    assert (_g.dst == g.dst).all()

def test_graph_index_jit():
    from galax.graph_index import GraphIndex
    g = GraphIndex(
        2,
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array([0, 2], dtype=jnp.int32)
    )

    @jax.jit
    def fn(g):
        g = g._replace(src=g.src+g.dst)
        return g

    _g = fn(g)
    assert _g.n_nodes == g.n_nodes
    assert (_g.dst == g.dst).all()

def test_heterograph_index_pytree():
    from galax.graph_index import GraphIndex
    from galax.heterograph_index import HeteroGraphIndex
    metagraph = GraphIndex(3, jnp.array([0, 1]), jnp.array([1, 2]))
    n_nodes = jnp.array([3, 2, 1])
    edges = ((jnp.array([0, 1]), jnp.array([1, 2])), (), ())
    g = HeteroGraphIndex(
         metagraph=metagraph, n_nodes=n_nodes, edges=edges,
    )
    children, aux_data = g.tree_flatten()
    _g = g.tree_unflatten(aux_data, children)
    assert _g.metagraph.n_nodes == g.metagraph.n_nodes
    assert (_g.metagraph.src == g.metagraph.src).all()
    assert (_g.metagraph.dst == g.metagraph.dst).all()
    assert (g.n_nodes == _g.n_nodes).all()

def test_heterograph_index_jit():
    from galax.graph_index import GraphIndex
    from galax.heterograph_index import HeteroGraphIndex
    metagraph = GraphIndex(3, jnp.array([0, 1]), jnp.array([1, 2]))
    n_nodes = jnp.array([3, 2, 1])
    edges = ((jnp.array([0, 1]), jnp.array([1, 2])), (), ())
    g = HeteroGraphIndex(
         metagraph=metagraph, n_nodes=n_nodes, edges=edges,
    )


    @jax.jit
    def fn(g):
        g._replace(n_nodes = g.n_nodes ** 2)
        return g

    _g = fn(g)
