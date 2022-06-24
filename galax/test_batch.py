import galax
from galax.batch import pad
g = galax.graph(([0, 0, 2], [0, 1, 2]))
_g = pad(g, g.number_of_nodes(), g.number_of_edges())
print(_g == g)
