import logging, argparse, math
from multiprocessing import Pool
from operator import itemgetter


def read_adj_graph(f):
    inlinks, outlinks = {}, {}
    
    for line in f:
        if len(line.strip()) == 0:
            continue
        elif '#' in line:
            tokens = line[:line.find('#')].split()
        else:
            tokens = line.split()

        if len(tokens) == 0:
            continue
        
        src = tokens[0].strip()
        for neighbor in tokens[1:]:
            dest = neighbor.strip()
            
            if dest not in inlinks:
                inlinks[dest] = set()
            inlinks[dest].add(src)

            if src not in outlinks:
                outlinks[src] = set()
            outlinks[src].add(dest)

    return inlinks, outlinks


def chunks(l, n):
    '''Yield successive n-sized chunks from l.'''
    for i in range(0, len(l), n):
        yield l[i:i + n]


def nodes_pagerank(nodes, N, old_pageranks, dangling_sum):
    global inlinks, outlinks, d

    prs = {}
    for node in nodes:
        nghbr_sum = 0
        if node in inlinks:
            for nghbr in inlinks[node]:
                nghbr_sum += old_pageranks[nghbr] / len(outlinks[nghbr])
        nghbr_sum += dangling_sum
        pr = (1 - d) / N + d * nghbr_sum
        prs[node] = pr
    return prs


def mapper(args):
    return nodes_pagerank(*args)


def pagerank_iteration(old_pageranks):
    global inlinks, outlinks, nodes, no_outlinks_nodes, d, num_threads

    # this is the sum that needs to be added to each node for PR to equal to 1
    # needs to be recomputed each iteration, before computing each node's PR
    dangling_sum = 0
    for node in no_outlinks_nodes:
        dangling_sum += old_pageranks[node] / len(nodes)
    
    if num_threads == 1:
        prs = nodes_pagerank(nodes, len(nodes), old_pageranks, dangling_sum)
    elif num_threads > 1:
        data = []
        for chunk in chunks(nodes, math.ceil(len(nodes) / num_threads)):
            data.append((chunk, len(nodes), old_pageranks, dangling_sum))

        pool = Pool(processes=num_threads)
        results = pool.map(mapper, data)
        prs = {}
        for r in results:
            prs.update(r)
    else:
        raise ValueError('Invalid number of threads.')

    return prs


def pagerank():
    global inlinks, outlinks, d, target_delta, num_threads

    new_prs = {node: 1 / len(nodes) for node in nodes}

    delta = target_delta + 1
    iter_num = 1
    while delta > target_delta:
        logging.debug('Starting iteration %d.' % iter_num)
        
        old_prs = new_prs
        new_prs = pagerank_iteration(old_prs)
        delta = sum([abs(old_prs[node] - new_prs[node]) for node in nodes])
        iter_num += 1

        logging.debug("New delta: %.9f" % delta)
        logging.debug("Pagerank sum: %.2f" % sum(new_prs.values())) # should always be 1

    return new_prs


def undirected_graph():
    global inlinks, outlinks
    links = {}

    def add_links(links, links_to_add):
        for node1, assoc_nodes in links_to_add.items():
            for node2 in assoc_nodes:
                if node1 not in links:
                    links[node1] = set()
                links[node1].add(node2)
                if node2 not in links:
                    links[node2] = set()
                links[node2].add(node1)

    add_links(links, inlinks)
    add_links(links, outlinks)
    return links


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute the pageranks of the nodes in a directed, unweighted graph.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('graph_file', type=str, help='The path to a graph in NetworkX adjacency list format.')
    parser.add_argument('--damping', type=float, default=.85, help='The damping factor.')
    parser.add_argument('--delta', type=float, default=10 ** -4, help='The cumulative change in PR between two iterations that is acceptable for terminating the algorithm. The smaller the number, the better convergence, but the more iterations it will take.')
    parser.add_argument('--num_threads', type=int, default=1, help='The number of threads to use for the computation.')
    parser.add_argument('--undirected', action='store_true', help='Treat the graph as undirected.')
    parser.add_argument('--sort', action='store_true', help='Store the output by pagerank in decreasing order.')
    parser.add_argument('--debug', action='store_true', help='Display debug messages.')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    return args


def write_output(prs, sort):
    logging.debug('Pagerank sum: %f' % sum(prs.values()))
    if sort:
        output = sorted(prs.items(), key=itemgetter(1), reverse=True)
    else:
        output = prs.items()
        
    for node, pr in output:
        print('%s\t%f' % (node, pr))
        pass


inlinks, outlinks = {}, {}
no_outlinks_nodes = set()
nodes = []
d, num_threads, target_delta = 0, 0, 0

if __name__ == '__main__':
    args = parse_args()

    with open(args.graph_file, 'r') as f:
        inlinks, outlinks = read_adj_graph(f)
    if args.undirected:
        links = undirected_graph(inlinks, outlinks)
        inlinks = links
        outlinks = links

    nodes = set(inlinks.keys()).union(set(outlinks.keys()))
    no_outlinks_nodes = nodes - set(outlinks.keys())
    nodes = list(nodes) # convert to a list to preserve the order

    d = args.damping
    target_delta = args.delta
    num_threads = args.num_threads

    prs = pagerank()

    write_output(prs, args.sort)
