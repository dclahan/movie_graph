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

with open('save_vars.pickle', 'rb') as f:
	g,ID_title_dict = pickle.load(f)

actors = {n for n, d in g.nodes(data=True) if d['bipartite'] == 1}
movies = set(g) - actors

# 2 hop neighborhoods of actors
def two_hop(x, visited):
	ret = set()
	for n in g.neighbors(x):
		ret = ret | set([p for p in g.neighbors(n) if visited[p] == 0])
	return ret - set([x])

# get new graph
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


with open('save_vars_weighted.pickle', 'rb') as f:
	G = pickle.load(f)

# G = initializeG()
# print()
# print(G.order())
# print(G.size())
# print()

# closeness centrality?? w weighted edges n stuff