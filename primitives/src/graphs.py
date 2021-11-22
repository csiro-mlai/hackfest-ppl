
from graphviz import Digraph



def coin_graph():
    """
    Generates a graph of the coin problem.

    :param n: The number of coins.
    :return: A graphviz Digraph object.
    """
    g = Digraph(comment='Coin problem')
    g.attr('node', shape='circle')
    g.node('fairness')
    g.node('toss', style='filled')
    g.edge('fairness', 'toss')
    return g

def coin_graph_multi(n):
    """
    Generates a graph of the coin problem, expanded.

    :param n: The number of coins.
    :return: A graphviz Digraph object.
    """
    g = Digraph(comment='Coin problem')
    g.attr('node', shape='circle')
    g.node('fairness')
    for i in range(n):
        tossname = f"toss_{i}"
        g.node(tossname, style='filled')
        g.edge('fairness', tossname)
    return g


def coin_graph_plate(n):
    """
    Generates a graph of the coin problem, as a plate.

    :param n: The number of coins.
    :return: A graphviz Digraph object.
    """
    g = Digraph(comment='Coin problem')
    g.attr('node', shape='circle')
    g.node('fairness')
    with g.subgraph(name='cluster_1') as c:
        c.attr(color='black', label=str(n), labelloc="b")
        c.node('toss', style='filled') 
    g.edge('fairness', 'toss')
    return g


def ruggedness_graph(n):
    """
    Generates a graph of the ruggedness regression, as a plate.

    :param n: The number of coins.
    :return: A graphviz Digraph object.
    """
    g = Digraph(comment='Ruggedness/GDP regression')
    g.attr('node', shape='circle')

    with g.subgraph(name='cluster_coeffs') as c:
        c.attr(color='none')
        c.node("a")
        c.node("bA")
        c.node("bR")
        c.node("bAR")
        c.node("sigma", "δ")

    with g.subgraph(name='cluster_1') as c:
        c.attr(color='black', label=str(n))
        c.node('ruggedness', style='filled')
        c.node('in Africa', style='filled')
        c.node('GDP', style='filled')

    g.edge("a", "GDP")
    g.edge("bA", "GDP")
    g.edge("bR", "GDP")
    g.edge("bAR", "GDP")
    g.edge("sigma", "GDP")
    g.edge("ruggedness", "GDP")
    g.edge("in Africa", "GDP")

    return g
