from functools import partial
import jax
from flax import linen as nn
import optax
import galax
from importlib import import_module

def run(args):
    from galax.data.datasets.nodes.graphsage import reddit
    G = reddit()
    OUT_FEATURES = Y_RED.max() + 1
    Y_REF = jax.nn.one_hot(G.ndata['label'], OUT_FEATURES)

    from galax.nn.zoo.graphsage import GraphSAGE
    model = nn.Sequential(
        GraphSAGE(args.features),
    )




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run(args)
