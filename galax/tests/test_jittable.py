import jax
import jax.numpy as jnp

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
        _g = g._replace(n_nodes = g.n_nodes ** 2)
        return _g

    _g = fn(g)
    assert (_g.n_nodes == g.n_nodes ** 2).all()

def test_graph_jit():
    import galax
    import jax
    import jax.numpy as jnp
    g = galax.graph(((0, 1), (1, 2)))
    g = g.set_ndata("h", jnp.ones(3))
    g = g.set_edata("he", jnp.ones(2))

    @jax.jit
    def fn(g):
        return g.edata["he"] ** 2

    fn(g)


def test_message_passing_jit():
    import galax
    import jax
    import jax.numpy as jnp
    g = galax.graph(((0, 1), (1, 2)))
    g = g.set_ndata("h", jnp.ones(3))

    @jax.jit
    def fn(g):
        mfunc = galax.function.copy_u("h", "m")
        rfunc = galax.function.sum("m", "h1")
        _g = galax.message_passing(g, mfunc, rfunc)
        return _g

    _g = fn(g)
    print(_g.ndata)
