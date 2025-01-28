import numpy as np
import itertools

num_genes = 3

"""
if no self-loops:
n=1: 1
n=2: 2
n=3: 6
n=4: 42



Code forward euler method!
"""

class Graph():
    def __init__(self, nodes):
        self.adj = np.zeros((nodes, nodes))

    def add_edge(self, source, target):
        self.adj[source,target] = 1

    def del_edge(self, source, target):
        self.adj[source,target] = -1

    def has_edge(self, source, target):
        return self.adj[source,target] == 1

    def __eq__(self, other_graph):
        if self.adj.shape != other_graph.adj.shape:
            return False
        n = self.adj.shape[0]  # Number of vertices

        for perm in itertools.permutations(range(n)):
            perm_matrix = np.eye(n)[list(perm)]
            permuted_matrix = np.dot(perm_matrix.T, np.dot(self.adj, perm_matrix))
            if np.array_equal(permuted_matrix, other_graph.adj):
                return True
        return False
    
self_loops_allowed = True

    
distinct_graphs = []
for mask in range(3 ** (num_genes*num_genes)):
    curr_graph = Graph(num_genes)
    for source in range(num_genes):
        for target in range(num_genes):
            if(mask // (3**(source*num_genes+target)) % 2 == 1):
                if(not self_loops_allowed):
                    if(source == target):
                        continue
                curr_graph.add_edge(source, target)
            elif(mask // (3**(source*num_genes+target)) %3 == 2):
                if(not self_loops_allowed):
                    if(source == target):
                        continue
                curr_graph.del_edge(source, target)
    for i in distinct_graphs:
        if(curr_graph == i):
            break
    else:
        distinct_graphs.append(curr_graph)

for i in distinct_graphs:
    print(i.adj, "\n\n\n")