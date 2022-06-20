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
