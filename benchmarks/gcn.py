from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import galax

def run(args):
    from galax.data.datasets.nodes.planetoid import cora
    G = cora()
    Y_REF = jax.nn.one_hot(G.ndata['label'], 7)

    from galax.nn.zoo.gcn import GCN
    model = galax.nn.Sequential(
        (
            galax.ApplyNodes(nn.Dropout(0.5, deterministic=False)),
            GCN(args.features),
            galax.ApplyNodes(nn.Dropout(0.5, deterministic=False)),
            GCN(7),
        ),
    )

    model_eval = galax.nn.Sequential(
        (
            galax.ApplyNodes(nn.Dropout(0.5, deterministic=True)),
            GCN(args.features),
            galax.ApplyNodes(nn.Dropout(0.5, deterministic=True)),
            GCN(7),
        ),
    )


    key = jax.random.PRNGKey(2666)
    key, key_dropout = jax.random.split(key)

    params = model.init(
        {"params": key, "dropout": key_dropout},
        G,
        method=model.uno,
    )

    optimizer = optax.chain(
        optax.adam(1e-2),
        optax.additive_weight_decay(5e-4),
    )

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
        y = g.ndata['h'].argmax(-1)
        accuracy_vl = (Y_REF[g.ndata['val_mask']].argmax(-1) ==
                y[g.ndata['val_mask']]).sum() / g.ndata['val_mask'].sum()
        accuracy_te = (Y_REF[g.ndata['test_mask']].argmax(-1) ==
                y[g.ndata['test_mask']]).sum() / g.ndata['test_mask'].sum()
        return accuracy_vl, accuracy_te

    import tqdm
    for _ in (pbar := tqdm.tqdm(range(500))):
        state, key = step(state, key)
        accuracy_vl, accuracy_te = eval(state)
        pbar.set_description(f"{accuracy_vl:.2f}, {accuracy_te:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=64)
    args = parser.parse_args()
    run(args)
