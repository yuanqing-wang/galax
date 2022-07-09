import ogb
import time
import numpy as onp
import jax
import jax.numpy as jnp
import galax

n_cold_start = 2

def bench_spmm(G, binary_op, reduce_op):
    print("SPMM\n----------------------------")

    for n_hid in [1, 2, 4, 8, 16, 32, 64, 128]:
        nfeat = jnp.array(onp.random.normal(size=(G.number_of_nodes(), n_hid)))
        efeat = jnp.array(onp.random.normal(size=(G.number_of_edges(), n_hid)))
        g = G.ndata.set("h", nfeat)
        g = G.edata.set("h", efeat)
        accum_time = 0

        @jax.jit
        def fn(g):
            fn_msg = getattr(galax.function, binary_op)("h", "m")
            fn_rdc = getattr(galax.function, reduce_op)("m", "h")
            _g = g.update_all(fn_msg, fn_rdc)
            return _g.ndata['h']

        for n_times in range(10):
            time0 = time.time()
            _g = fn(G).block_until_ready()
            time1 = time.time()
            if n_times >= n_cold_start:
                accum_time += (time1 - time0)
        avg_time = accum_time / (n_times - n_cold_start)
        print('hidden size: {}, avg time: {}'.format(
            n_hid, avg_time))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Benchmark DGL kernels")
    parser.add_argument('--spmm-binary', type=str, default='copy_u')
    parser.add_argument('--spmm-reduce', type=str, default='sum')

    args = parser.parse_args()

    # for dataset in ['reddit', 'arxiv', 'proteins']:
    from galax.data.datasets.nodes.ogb import arxiv
    G = arxiv()
    print(G)

    # SPMM
    bench_spmm(G, args.spmm_binary, args.spmm_reduce)
