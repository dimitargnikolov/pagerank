# pagerank

Compute the PageRanks of the nodes in a directed, unweighted graph, using Python's `multiprocessing` API. 

Example usage:

```
$ python pagerank.py dummy-data/network1.adj > pageranks.out
```

For other usage details, do:

```
$ python pagerank.py --help
```

The `nx_pagerank.py` is included for testing purposes. It uses networkx's PageRank implementation.

To generate more substantial testing data, do something like this:

```
>>> import networkx as nx
>>> g = nx.barabasi_albert_graph(10**5, 50)
>>> nx.write_adjlist(g, 'dummy-data/network4-ba-n100000-m50.adj')
```
