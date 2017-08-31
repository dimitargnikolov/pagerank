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
