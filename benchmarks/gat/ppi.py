"""Reference: 81.5; Reproduction: 81.8"""

from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import galax


def run():
    from galax.data.datasets.nodes.graphsage import ppi
    GS_TR, GS_VL, GS_TE = ppi()
    GS_TR = tuple(map(lambda g: g.add_self_loop(), GS_TR))
    GS_VL = tuple(map(lambda g: g.add_self_loop(), GS_VL))
    GS_TE = tuple(map(lambda g: g.add_self_loop(), GS_TE))

    from galax.data.dataloader import PrixFixeDataLoader
    ds_tr = PrixFixeDataLoader(GS_TR, 2)
    g_vl = galax.batch(GS_VL)
    g_te = galax.batch(GS_TE)

    from galax.nn.zoo.gat import GAT
    ConcatenationPooling = galax.ApplyNodes(lambda x: x.reshape(*x.shape[:-2], -1))
    AveragePooling = galax.ApplyNodes(lambda x: x.mean(-2))

    model = galax.nn.Sequential(
        (
            GAT(256, 4, activation=jax.nn.elu),
            ConcatenationPooling,
            GAT(256, 4, activation=jax.nn.elu),
            ConcatenationPooling,
            GAT(121, 6, activation=None),
            AveragePooling,
        ),
    )

    key = jax.random.PRNGKey(2666)
    key, key_dropout = jax.random.split(key)

    params = model.init({"params": key, "dropout": key_dropout}, next(iter(ds_tr)))

    optimizer = optax.adam(0.005)

    from flax.training.train_state import TrainState
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    @jax.jit
    def loss(params, g):
        g = model.apply(params, g)
        _loss = optax.sigmoid_binary_cross_entropy(
            g.ndata['h'], g.ndata['label'],
        ).mean()
        # _loss = jnp.where(jnp.expand_dims(g.is_not_dummy(), -1), _loss, 0.0)
        # _loss = _loss.sum() / len(_loss)
        return _loss

    @jax.jit
    def step(state, g):
        grad_fn = jax.grad(partial(loss, g=g))
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    def eval(state, g):
        g = model.apply(params, g)
        y_hat = g.ndata['h'][g.is_not_dummy()]
        y_hat = 1 * (jax.nn.sigmoid(y_hat) > 0.5)
        y = g.ndata['h'][g.is_not_dummy()]
        accuracy = (y_hat == y) / y.size
        return accuracy

    from galax.nn.utils import EarlyStopping
    early_stopping = EarlyStopping(10)

    import tqdm
    for _ in tqdm.tqdm(range(1000)):
        for idx, g in enumerate(ds_tr):
            _loss = loss(state.params, g)



if __name__ == "__main__":
    import argparse
    run()
