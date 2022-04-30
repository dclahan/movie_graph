import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import numpy as np

def draw_comm_graph(G, comms):
	colors = [comms[v] for v in G.nodes()]
	if G.order() < 50:
		nx.draw_kamada_kawai(G, node_color=colors)
		plt.show()
	else:
		nx.draw(G,node_size=25, node_color = colors)
		plt.show()

def label_prop(G):
	comms = {}
	for v in G.nodes():
		comms[v] = int(v)
	updates = 1
	while updates > 0:
		updates = 0
		for v in sorted(G.nodes(), key=lambda k: random.random()):
			counts = {}
			for u in G.neighbors(v):
				if comms[u] not in dict.keys(counts):
					counts[comms[u]] = 1
				else:
					counts[comms[u]] += 1
			c = np.random.choice([k for k in counts.keys() if counts[k]==max(counts.values())])
			if c != comms[v]:
				comms[v] = c
				updates += 1
	return comms

def label_prop_weighted(G, bias = 1):
	comms = {}
	for v in G.nodes():
		comms[v] = int(v)
	updates = 1
	while updates > 0:
		updates = 0
		for v in sorted(G.nodes(), key=lambda k: random.random()):
			counts = {}
			for u in G.neighbors(v):
				if comms[u] not in dict.keys(counts):
					counts[comms[u]] = bias*G[u][v]['weight']
				else:
					counts[comms[u]] += bias*G[u][v]['weight']
			c = np.random.choice([k for k in counts.keys() if counts[k]==max(counts.values())])
			if c != comms[v]:
				comms[v] = c
				updates += 1
	return comms


def edge_cut(G, comms):
	cut = 0
	for v in G.nodes():
		for u in G.neighbors(v):
			if comms[u] != comms[v]:
				cut += 1
	return cut

def conductance(G, comms):
	unique_comms = list(set([x for x in comms.values()]))
	avg_conductance = 0.0
	for c in unique_comms:
		cut = 0
		dsum = 0
		for v in G.nodes():
			if comms[v] == c:
				dsum += G.degree(v)
				for u in G.neighbors(v):
					if comms[v] != comms[u]:
						cut += 1
		dsum_comp = G.size()*2 - dsum
		avg_conductance += cut / min(dsum, dsum_comp)
	return avg_conductance / len(unique_comms)

def modularity(G, comms):
	unique_comms = list(set([x for x in comms.values()]))
	new_comms = []
	for c in unique_comms:
		C = []
		for v in G.nodes():
			if comms[v] == c:
				C.append(v)
		new_comms.append(set(C))
	return nx.community.modularity(G, new_comms)