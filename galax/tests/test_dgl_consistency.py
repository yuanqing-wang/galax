import pytest
import dgl
import galax
import numpy as onp
import jax.numpy as jnp

graphs = []
for n_nodes in [2, 10, 20]:
    for n_edges in range(n_nodes, n_nodes**2, n_nodes):
        graphs.append(dgl.rand_graph(n_nodes, n_edges))


tensor_shapes = [
    (3, ),
    (3, 3),
    (3, 3, 1),
    (1, 3, 1, 3),
    (1, 3, 1, 1, 3),
]

spmm_shapes = [
    ((1, 2, 1, 3, 1), (4, 1, 3, 1, 1)),
    ((3, 3), (1, 3)),
    ((1,), (3,)),
    ((3,), (1,)),
    ((1,), (1,)),
    ((), ())
]

@pytest.mark.parametrize("g", graphs)
def test_from_dgl(g):
    _g = galax.from_dgl(g)
    assert _g.number_of_nodes() == g.number_of_nodes()
    assert _g.number_of_edges() == g.number_of_edges()
    _src, _dst = _g.edges()
    src, dst = g.edges()
    assert(src.numpy().tolist() == _src.tolist())
    assert(dst.numpy().tolist() == _dst.tolist())

@pytest.mark.parametrize("g", graphs)
@pytest.mark.parametrize("shape", tensor_shapes)
def test_apply_nodes(g, shape):
    _g = galax.from_dgl(g)

    W = onp.random.normal(size=(shape[-1], 7))
    B = onp.random.normal(size=(7, ))
    X = onp.random.normal(size=(g.number_of_nodes(), *shape))

    import torch
    w = torch.tensor(W)
    b = torch.tensor(B)
    x = torch.tensor(X)
    _w = jnp.array(w)
    _b = jnp.array(b)
    _x = jnp.array(x)

    def fn(x):
        return x @ w + b

    def _fn(_x):
        return _x @ _w + _b

    _g = _g.ndata.set("h", _x)
    g.ndata["h"] = x

    _g = galax.apply_nodes(_fn)(_g)
    _y = onp.array(_g.ndata['h'])

    g.apply_nodes(lambda nodes: {'h': fn(nodes.data['h'])})
    y = g.ndata['h'].detach().numpy()

    assert onp.allclose(y, _y, rtol=1e-3)

@pytest.mark.parametrize("g", graphs)
@pytest.mark.parametrize("shape", tensor_shapes)
def test_apply_edges(g, shape):
    _g = galax.from_dgl(g)

    W = onp.random.normal(size=(shape[-1], 7))
    B = onp.random.normal(size=(7, ))
    X = onp.random.normal(size=(g.number_of_edges(), *shape))

    import torch
    w = torch.tensor(W)
    b = torch.tensor(B)
    x = torch.tensor(X)
    _w = jnp.array(w)
    _b = jnp.array(b)
    _x = jnp.array(x)

    def fn(x):
        return x @ w + b

    def _fn(_x):
        return _x @ _w + _b

    _g = _g.edata.set("h", _x)
    g.edata["h"] = x

    _g = galax.apply_edges(_fn)(_g)
    _y = onp.array(_g.edata['h'])

    g.apply_edges(lambda edges: {'h': fn(edges.data['h'])})
    y = g.edata['h'].detach().numpy()

    assert onp.allclose(y, _y, rtol=1e-2)


@pytest.mark.parametrize('g', graphs)
@pytest.mark.parametrize('shp', spmm_shapes)
@pytest.mark.parametrize('msg', ['add', 'sub', 'mul', 'div', 'copy_lhs'])
@pytest.mark.parametrize('reducer', ['sum', 'min', 'max'])
def test_message_passing(g, shp, msg, reducer):
    _g = galax.from_dgl(g)

    udf_msg_dgl = {
        'add': lambda edges: {'m': edges.src['x'] + edges.data['w']},
        'sub': lambda edges: {'m': edges.src['x'] - edges.data['w']},
        'mul': lambda edges: {'m': edges.src['x'] * edges.data['w']},
        'div': lambda edges: {'m': edges.src['x'] / edges.data['w']},
        'copy_lhs': lambda edges: {'m': edges.src['x']},
        'copy_rhs': lambda edges: {'m': edges.data['w']},
    }

    import torch
    udf_reduce_dgl = {
        'sum': lambda nodes: {'v': torch.sum(nodes.mailbox['m'], 1)},
        'min': lambda nodes: {'v': torch.min(nodes.mailbox['m'], 1)[0]},
        'max': lambda nodes: {'v': torch.max(nodes.mailbox['m'], 1)[0]}
    }

    hu = onp.random.normal(size=(g.number_of_nodes(),) + shp[0]) + 1.0
    he = onp.random.normal(size=(g.number_of_edges(),) + shp[1]) + 1.0

    import torch
    hu_dgl = torch.tensor(hu)
    he_dgl = torch.tensor(he)
    g.ndata['x'] = hu_dgl
    g.edata['w'] = he_dgl

    hu_glx = jnp.array(hu)
    he_glx = jnp.array(he)
    _g = _g.ndata.set('x', hu)
    _g = _g.edata.set('w', he)

    g.update_all(udf_msg_dgl[msg], udf_reduce_dgl[reducer])

    if msg == "copy_lhs":
        msg_fn_glx = galax.function.copy_u("x", "m")
    elif msg == "copy_rhs":
        msg_fn_glx = galax.function.copy_e("w", "m")
    else:
        msg_fn_glx = getattr(galax.function, f"u_{msg}_e")("x", "w", "m")
    reduce_fn_glx = getattr(galax.function, reducer)("m", "v")
    _g = _g.update_all(msg_fn_glx, reduce_fn_glx)

    y_dgl = g.ndata['v'].detach().numpy()
    y_glx = onp.array(_g.ndata['v'])

    assert onp.allclose(y_dgl, y_glx, rtol=1e-2)
