import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import random
import pickle
import sys

with open('save_vars.pickle', 'rb') as f:
	g,ID_title_dict = pickle.load(f)

actors = {n for n, d in g.nodes(data=True) if d['bipartite'] == 1}
movies = set(g) - actors

title_ID_dict = dict([(tit,ID) for ID,tit in ID_title_dict.items()]) # doesnt account for collisions

def print_sp(sp):
	if not sp:
		return
	for n in sp:
		s = ID_title_dict[str(n)] if str(n) in ID_title_dict.keys() else n
		if n == sp[-1]:
			print(f"{s}")
			continue
		print(f"{s} ", end = "=> ")

def shortest_paf(source,target): # doesnt account for collisions
       if source not in actors:
              if source not in title_ID_dict.keys():
                     print(f"{source} not found")
                     return
              source = int(title_ID_dict[source])
       if target not in actors:
              if target not in title_ID_dict.keys():
                     print(f"{target} not found")
                     return 
              target = int(title_ID_dict[target])

       sp = nx.shortest_path(g, source=source, target=target)
       return sp

if __name__ == "__main__":
	if len(sys.argv) == 3:
		print()
		sp = shortest_paf(sys.argv[1],sys.argv[2])
		print_sp(sp)
		print()
		# path = g.subgraph(sp)
		# nx.draw(path, with_labels=True)
		# plt.show()
	elif len(sys.argv) == 2:
		nodes = list(g.nodes())
		for i in range(int(sys.argv[1])):
			root = random.choice(nodes)
			paths = nx.single_source_shortest_path(g,root, cutoff = 10)
			path = max(paths.values(), key=lambda x: len(x))
			print_sp(path)
	else:
		print("please give 2 nodes to find path between")
