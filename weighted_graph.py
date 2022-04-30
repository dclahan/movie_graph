import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import random
import math
import operator
from itertools import chain
from tqdm import tqdm
import pandas as pd
import numpy as np
import ast
import pickle
import time
from module import Movie

# Make sure you've initiated the graph in "project.py"
with open('save_vars.pickle', 'rb') as f:
	g,ID_title_dict = pickle.load(f)

actors = {n for n, d in g.nodes(data=True) if d['bipartite'] == 1}
movies = set(g) - actors

# FUNCTIONS
def two_hop(x, visited):
	ret = set()
	for n in g.neighbors(x):
		ret = ret | set([p for p in g.neighbors(n) if visited[p] == 0])
	return ret - set([x])

def shared(x,y):
	return len(set(g.neighbors(x)) & set(g.neighbors(y)))

def initializeG():
	G = nx.Graph()
	visited_movs = dict([(id,0) for id in movies])
	root = random.choice(list(movies))
	Q = [root]
	for _ in tqdm(range(len(movies)), desc = "Creating G"):
		if len(Q) == 0:
			print("Made G!")
			break
		curr = Q.pop(0)
		visited_movs[curr] = 1
		two_hops = two_hop(curr, visited_movs)
		for mov in two_hops:
			G.add_edge(curr,mov,weight = shared(curr,mov)) 
			if mov not in Q:
				Q.append(mov)
	with open('save_vars_weighted.pickle', 'wb') as f:
		f.write(pickle.dumps(G))
	return G

def pl_ex_coef(degrees, order):
       xmin = min(degrees.keys())
       ahat = 1  +  (order)  *  (1/sum([v*math.log(k/xmin) for (k,v) in degrees.items()]))
       pl_coef = (ahat-1)/xmin
       return ahat, pl_coef

####################################################################################################################

with open('save_vars_weighted.pickle', 'rb') as f:
	G = pickle.load(f)

# G = initializeG() # UNCOMMENT TO CREATE/PICKLE THE GRAPH

# nx.write_graphml(G, 'wmovs.graphml')

print(G.order())
print(G.size())

# closeness centrality - takes WAY too long, too many edges with too many weights to be accessed
# most edge weights are just 1, so use...
# degree centrality instead:
cent = sorted(nx.degree_centrality(G).items(), reverse=True, key=lambda x : x[1])
print([(str(ID_title_dict[str(cent[i][0])]), cent[i][1]) for i in range(10)])

degs = {}
for n in G.nodes():
	d = G.degree(n)
	if d not in degs.keys():
		degs[d] = 1
	else:
		degs[d]+=1

sorteddegs = sorted(degs.items())
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([k for (k,v) in sorteddegs], [v for (k, v) in sorteddegs], 'b')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('degree')
ax.set_ylabel('frequency')
plt.show()


pl_exp, pl_coefficient = pl_ex_coef(degs,G.order())
print(pl_exp)


cutoff = math.log(G.order())
tail = [v for (k,v) in degs.items() if k > cutoff]
hub_ratio = sum(tail)/G.order()

print("hub ratio:", hub_ratio)
