"""Reference: 97.3; Reproduction: 97.4"""

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
    _ConcatenationPooling = lambda x: x.reshape(*x.shape[:-2], -1)
    ConcatenationPooling = galax.ApplyNodes(_ConcatenationPooling)
    _AveragePooling = lambda x: x.mean(-2)
    AveragePooling = galax.ApplyNodes(_AveragePooling)

    class Model(nn.Module):
        def setup(self):
            self.l0 = GAT(256, 4, activation=jax.nn.elu)
            self.l1 = GAT(256, 4, activation=jax.nn.elu)
            self.l2 = GAT(121, 6, activation=None)

        def __call__(self, g):
            g0 = ConcatenationPooling(self.l0(g))
            g1 = ConcatenationPooling(self.l1(g0))
            g1 = g1.ndata.set("h", g1.ndata['h'] + g0.ndata['h'])
            g2 = AveragePooling(self.l2(g1))
            return g2

    model = Model()

    key = jax.random.PRNGKey(2666)
    key, key_dropout = jax.random.split(key)

    params = model.init({"params": key, "dropout": key_dropout}, next(iter(ds_tr)))

    optimizer = optax.adam(0.005)

    from flax.training.train_state import TrainState
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    # @jax.jit
    def loss(params, g):
        g = model.apply(params, g)
        _loss = optax.sigmoid_binary_cross_entropy(
            g.ndata['h'], g.ndata['label'],
        ).mean()
        _loss = jnp.where(jnp.expand_dims(g.is_not_dummy(), -1), _loss, 0.0)
        _loss = _loss.sum() / len(_loss)
        return _loss

    @jax.jit
    def step(state, g):
        grad_fn = jax.grad(partial(loss, g=g))
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    # @jax.jit
    def eval(state, g):
        g = model.apply(state.params, g)
        _loss = optax.sigmoid_binary_cross_entropy(
            g.ndata['h'], g.ndata['label'],
        ).mean()
        _loss = jnp.where(jnp.expand_dims(g.is_not_dummy(), -1), _loss, 0.0)
        _loss = _loss.sum() / len(_loss)

        y_hat = g.ndata['h']
        y_hat = 1 * (jax.nn.sigmoid(y_hat) > 0.5)
        y = g.ndata['label']

        y_hat = y_hat[g.is_not_dummy()]
        y = y[g.is_not_dummy()]
        accuracy = (y_hat == y).sum() / y.size

        return _loss, accuracy

    from galax.nn.utils import EarlyStopping
    early_stopping = EarlyStopping(100)

    import tqdm
    for _ in tqdm.tqdm(range(1000)):
        for idx, g in enumerate(ds_tr):
            state = step(state, g)
        loss_vl, accuracy_vl = eval(state, g_vl)
        if early_stopping((-accuracy_vl, loss_vl), state.params):
            state = state.replace(params=early_stopping.params)
            break

    _, accuracy = eval(state, g_te)
    print(accuracy)

if __name__ == "__main__":
    import argparse
    run()
