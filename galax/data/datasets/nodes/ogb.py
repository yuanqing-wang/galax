import galax

def arxiv():
    from ogb.graphproppred import DglGraphPropPredDataset
    g = DglGraphPropPredDataset(name="ogbn-arxiv")[0][0]
    g = galax.from_dgl(g)
    return g
