"""Reference: 79.0; Reproduction: 78.1"""

from functools import partial
import jax
from flax import linen as nn
import optax
import galax

def run():
    from galax.data.datasets.nodes.planetoid import pubmed
    G = pubmed()
    G = G.add_self_loop()
    Y_REF = jax.nn.one_hot(G.ndata['label'], 3)

    from galax.nn.zoo.gat import GAT
    ConcatenationPooling = galax.ApplyNodes(lambda x: x.reshape(*x.shape[:-2], -1))
    AveragePooling = galax.ApplyNodes(lambda x: x.mean(-2))

    model = nn.Sequential(
        (
            GAT(8, 8, attn_drop=0.4, feat_drop=0.4, deterministic=False, activation=jax.nn.elu),
            ConcatenationPooling,
            GAT(3, 8, attn_drop=0.4, feat_drop=0.4, deterministic=False, activation=None),
            AveragePooling,
        ),
    )

    model_eval = nn.Sequential(
        (
            GAT(8, 8, attn_drop=0.4, feat_drop=0.4, deterministic=True, activation=jax.nn.elu),
            ConcatenationPooling,
            GAT(3, 8, attn_drop=0.4, feat_drop=0.4, deterministic=True, activation=None),
            AveragePooling,
        ),
    )

    key = jax.random.PRNGKey(2666)
    key, key_dropout = jax.random.split(key)

    params = model.init({"params": key, "dropout": key}, G)
    mask = jax.tree_map(lambda x: (x != 0).any(), params)

    from flax.core import FrozenDict

    optimizer = optax.adam(0.01)

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

    _, __ = step(state, key)

    import time
    time0 = time.time()
    for _ in range(200):
        state, key = step(state, key)
    state = jax.block_until_ready(state)
    time1 = time.time()
    print(time1 - time0)

if __name__ == "__main__":
    import argparse
    run()
