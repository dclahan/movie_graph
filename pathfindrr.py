import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import random
import pickle
import sys
from module import Movie

# Make sure you've initiated the graph in "project.py"
# or importing <from project import initialize> i guess works too
with open('save_vars.pickle', 'rb') as f:
	g,ID_title_dict = pickle.load(f)

actors = {n for n, d in g.nodes(data=True) if d['bipartite'] == 1}
movies = set(g) - actors

title_ID_dict = dict()
for ID,mov in ID_title_dict.items():
	tit = mov.title
	if tit not in title_ID_dict.keys():
		title_ID_dict[tit] = [mov]
	else:
		title_ID_dict[tit].append(mov)

def print_sp(sp):
	if not sp:
		return
	for n in sp:
		s = ID_title_dict[str(n)] if str(n) in ID_title_dict.keys() else n
		if n == sp[-1]:
			print(f"{s}")
			continue
		print(f"{s} ", end = "=> ")

def shortest_paf(source,target):
       if source not in actors:
              if source not in title_ID_dict.keys():
                     print(f"{source} not found")
                     return
              src = title_ID_dict[source]
              inp = 0
              if len(src) > 1:
              	print(f"Source: Which {source} did you mean?")
              	for i in range(len(src)):
              		print("({0}):\t {1} ({2})".format(i,src[i].title.ljust(20), src[i].year))
              	inp = input(": ")
              	inp = int(inp)
              	while inp >= len(src):
              		inp = input(": ")
              		inp = int(inp)
              source = int(src[inp].ID)
              # print(source)
       if target not in actors:
              if target not in title_ID_dict.keys():
                     print(f"{target} not found")
                     return 
              targ = title_ID_dict[target]
              inp = 0
              if len(targ) > 1:
              	print(f"Target: Which {target} did you mean?")
              	for i in range(len(targ)):
              		print("({0}):\t {1} ({2})".format(i,targ[i].title.ljust(20), targ[i].year))
              	inp = input(": ")
              	inp = int(inp)
              	while inp >= len(targ):
              		inp = input(": ")
              		inp = int(inp)
              target = int(targ[inp].ID)
              # print(target)

       sp = nx.shortest_path(g, source=source, target=target)
       return sp

if __name__ == "__main__":
	if len(sys.argv) == 3:
		print()
		sp = shortest_paf(sys.argv[1],sys.argv[2])
		print_sp(sp)
		print()
	elif len(sys.argv) == 2:
		# Experimentally find hubs by the amount of times a vertex appears in a shortest path out of n random shortest paths 
		nodes = list(g.nodes())
		hubs = {}
		for i in range(int(sys.argv[1])):
			root = random.choice(nodes)
			target = random.choice(nodes)
			try:
				sp = nx.shortest_path(g, source=root, target=target)
			except:
				continue
			for n in sp:
				if n in hubs:
					hubs[n] += 1
				else:
					hubs[n] = 1
		total_visits = sum(hubs.values())
		sorted_hubs = sorted(list(hubs.items()), key = lambda x: x[1], reverse = True)
		hubby_movs = [x for x in sorted_hubs if x[0] in movies]
		print(f"From simulation of {sys.argv[1]} random shortest paths, we find the top 10 \nmovie hubs:")
		for i in range(10):
			print(ID_title_dict[str(hubby_movs[i][0])], hubby_movs[i][1]/total_visits)
		hubby_acts = [x for x in sorted_hubs if x[0] in actors]
		print("\nactor hubs:")
		for i in range(10):
			print(hubby_acts[i][0], hubby_acts[i][1]/total_visits)
	else:
		print("please give 2 nodes to find path between \n Or give the number of shortest paths to run our exerimental hub finder on")
