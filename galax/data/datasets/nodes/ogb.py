def arxiv():
    from ogb.nodeproppred import DglNodePropPredDataset
    import galax
    g = DglNodePropPredDataset(name="ogbn-arxiv")[0][0]
    g.ndata['h'] = g.ndata['feat']
    del g.ndata['feat']
    g = galax.from_dgl(g)
    return g
