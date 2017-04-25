from __future__ import division, print_function
import argparse
import networkx as nx
from pagerank import parse_args, write_output


if __name__ == '__main__':
	args = parse_args()
	G = nx.DiGraph(nx.read_adjlist(args.graph_file))
	prs = nx.pagerank(G, alpha=args.damping)
	write_output(prs, args.sort)
	
