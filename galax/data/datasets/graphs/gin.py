"""Datasets used in How Powerful Are Graph Neural Networks?
(chen jun)
Datasets include:
MUTAG, COLLAB, IMDBBINARY, IMDBMULTI, NCI1, PROTEINS,
PTC, REDDITBINARY, REDDITMULTI5K
https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip
"""
import sys

DATASETS = [
    "MUTAG", "COLLAB", "IMDBBINARY", "IMDBMULTI", "NCI1", "PROTEINS",
    "PTC", "REDDITBINARY", "REDDITMULTI5K",
]

def get_dataset_function(dataset):
    def transform(g, y):
        import galax
        g.ndata['h'] = g.ndata['attr']
        del g.ndata['h']
        g = galax.from_dgl(g)
        g = g.gdata.set("label", y)
        return g

    def fn():
        import dgl
        from dgl.data import GINDataset
        import galax
        _dataset = GINDataset(dataset, self_loop=False)
        gs, ys = zip(*[_dataset[idx] for idx in range(len(_dataset))])
        gs = tuple([transform(g, y) for g, y in zip(gs, ys)])
        return gs

    fn.__name__ = dataset.lower()
    return fn

for dataset in DATASETS:
    fn = get_dataset_function(dataset)
    setattr(sys.modules[__name__], dataset.lower(), fn)
