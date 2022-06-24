"""Reference: 81.5; Reproduction: 81.8"""

from functools import partial
import jax
from flax import linen as nn
import optax
import galax


def run():
    from galax.data.datasets.nodes.planetoid import ppi
    GS_TR, GS_VL, GS_TE = ppi()
    GS_TR = map(lambda g: g.add_self_loop(), GS_TR)
    GS_VL = map(lambda g: g.add_self_loop(), GS_VL)
    GS_TE = map(lambda g: g.add_self_loop(), GS_TE)

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

    params = model.init({"params": key, "dropout": key_dropout}, G)

    from flax.core import FrozenDict
    mask = FrozenDict(
        {"params":
            {
                "layers_1": True,
                "layers_3": False,
            },
        },
    )

    optimizer = optax.chain(
        optax.additive_weight_decay(5e-4, mask=mask),
        optax.adam(1e-2),
    )

    from flax.training.train_state import TrainState
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    def loss(params, key):
        g = model.apply(params, G, rngs={"dropout": key})
        y = g.ndata['h']
        return optax.softmax_cross_entropy(
            y[g.ndata['train_mask']],
            Y_REF[g.ndata['train_mask']],
        ).mean()

    @jax.jit
    def step(state, key):
        key, new_key = jax.random.split(key)
        grad_fn = jax.grad(partial(loss, key=new_key))
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, key

    @jax.jit
    def eval(state):
        params = state.params
        g = model_eval.apply(params, G)
        y = g.ndata['h']
        accuracy_vl = (Y_REF[g.ndata['val_mask']].argmax(-1) ==
                y[g.ndata['val_mask']].argmax(-1)).sum() /\
                g.ndata['val_mask'].sum()
        loss_vl = optax.softmax_cross_entropy(
            y[g.ndata['val_mask']],
            Y_REF[g.ndata['val_mask']],
        ).mean()
        return accuracy_vl, loss_vl

    @jax.jit
    def test(state):
        params = state.params
        g = model_eval.apply(params, G)
        y = g.ndata['h']
        accuracy_te = (Y_REF[g.ndata['test_mask']].argmax(-1) ==
            y[g.ndata['test_mask']].argmax(-1)).sum() /\
            g.ndata['test_mask'].sum()
        loss_te = optax.softmax_cross_entropy(
            y[g.ndata['test_mask']],
            Y_REF[g.ndata['test_mask']],
        ).mean()
        return accuracy_te, loss_te

    from galax.nn.utils import EarlyStopping
    early_stopping = EarlyStopping(10)

    import tqdm
    for _ in tqdm.tqdm(range(1000)):
        state, key = step(state, key)
        accuracy_vl, loss_vl = eval(state)
        if early_stopping((-accuracy_vl, loss_vl), state.params):
            state = state.replace(params=early_stopping.params)
            break

    accuracy_te, _ = test(state)
    print(accuracy_te)

if __name__ == "__main__":
    import argparse
    run()
