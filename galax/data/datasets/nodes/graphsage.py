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
