import pytest

def test_prefetch():
    import galax
    g = galax.graph({
        ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1])
    })
    g = g.remove_nodes([0, 1], ntype='game')
