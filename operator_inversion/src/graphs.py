
from graphviz import Digraph


def fno_graph(obs="fwd"):
    """
    Generates a graph of the operator inversion problem.
    """
    g = Digraph(comment='FNO operator inversion')
    g.attr('node', shape='circle')

    with g.subgraph(name='cluster_coeffs') as c:
        c.attr(color='none')
        c.node("xt-2", label="x(t-2)", style='filled')
        c.node("xt-1", label="x(t-1)", style='filled')
        c.node("xt", label="x(t)", style='filled')
    
    g.node("θ", style='filled' if obs!="fwd" else '')
    g.node("forcing", style='filled' if obs=="fwd" else '')
    g.edge("xt-2", "xt")
    g.edge("xt-1", "xt")
    g.edge("forcing", "xt")
    g.edge("θ", "xt")
    return g

