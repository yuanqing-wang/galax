"""Reddit and PPI.

Reference: http://snap.stanford.edu/graphsage/
"""

import galax

def reddit():
    from dgl.data import RedditDataset
    g = RedditDataset()[0]
    g.ndata['h'] = g.ndata['feat']
    del g.ndata['feat']
    g = galax.from_dgl(g)
    return g

def ppi():
    from dgl.data import PPIDataset
    gs_tr = PPIDataset("train")
    gs_vl = PPIDataset("valid")
    gs_te = PPIDataset("test")

    def fn(g):
        g.ndata['h'] = g.ndata['feat']
        del g.ndata['feat']
        return galax.from_dgl(g)

    gs_tr = tuple((fn(g) for g in gs_tr))
    gs_vl = tuple((fn(g) for g in gs_vl))
    gs_te = tuple((fn(g) for g in gs_te))
    return gs_tr, gs_vl, gs_te
