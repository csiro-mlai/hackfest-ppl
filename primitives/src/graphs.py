
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
        c.attr(color='black', label=str(n))
        c.node('toss', style='filled') 
    g.edge('fairness', 'toss')
    return g

