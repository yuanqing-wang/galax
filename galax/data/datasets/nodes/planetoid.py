"""Cora, citeseer, pubmed dataset.

Following dataset loading and preprocessing code from tkipf/gcn
https://github.com/tkipf/gcn/blob/master/gcn/utils.py
"""
import galax

def cora():
    from dgl.data import CoraGraphDataset
    g = CoraGraphDataset()[0]
    g.ndata['h'] = g.ndata['feat']
    del g.ndata['feat']
    g = galax.from_dgl(g)
    return g

def citeseer():
    from dgl.data import CiteseerGraphDataset
    g = CiteseerGraphDataset()[0]
    g.ndata['h'] = g.ndata['feat']
    del g.ndata['feat']
    g = galax.from_dgl(g)
    return g

def pubmed():
    from dgl.data import PubmedGraphDataset
    g = PubmedGraphDataset()[0]
    g.ndata['h'] = g.ndata['feat']
    del g.ndata['feat']
    g = galax.from_dgl(g)
    return g
