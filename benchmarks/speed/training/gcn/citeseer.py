"""Reference: 70.3; Reproduction: 71.4"""

from functools import partial
import jax
from flax import linen as nn
import optax
import galax


def run():
    from galax.data.datasets.nodes.planetoid import citeseer
    G = citeseer()
    G = G.add_self_loop()
    Y_REF = jax.nn.one_hot(G.ndata['label'], 6)

    from galax.nn.zoo.gcn import GCN
    model = nn.Sequential(
        (
            GCN(16, activation=jax.nn.relu, dropout=0.5, deterministic=False),
            GCN(6, activation=None, dropout=0.5, deterministic=False),
        ),
    )

    model_eval = nn.Sequential(
        (
            GCN(16, activation=jax.nn.relu, dropout=0.5, deterministic=False),
            GCN(6, activation=None, dropout=0.5, deterministic=False),
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

    optimizer = optax.adam(1e-2)

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
    def steps(state, key):
        for _ in range(10):
            state, key = step(state, key)
        return state, key

    for _ in range(2):
        _, __ = jax.block_until_ready(steps(state, key))

    import time
    time0 = time.time()
    for _ in range(20):
        state, key = steps(state, key)
    state = jax.block_until_ready(state)
    time1 = time.time()
    print(time1 - time0)

if __name__ == "__main__":
    import argparse
    run()
