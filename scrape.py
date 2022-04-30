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
from comm import * # community detection/visulaization algos

import requests
from bs4 import BeautifulSoup

## letterboxd scraper
def scrapey(user, G, title_ID_dict):
	url = f"https://letterboxd.com/{user}/films/by/release-earliest/page/1/"

	try:
		r = requests.get(url)
		print(r)
	except:
		print(f'could not open <{url}>')
		return nx.Graph()

	#find how many pages of movies make up the users "watched movies"
	soup = BeautifulSoup(r.content,'html.parser')
	s = soup.find('div', class_="site-body")
	um = s.find('div', class_="paginate-pages")
	aa = um.find_all('a')
	num_pages = int(aa[-1].get_text())


	persons_movies = []
	for i in tqdm(range(1,num_pages+1)):
		url = f"https://letterboxd.com/{user}/films/by/release-earliest/page/{i}/"
		try:
			r = requests.get(url)
			# print(r)
		except:
			print(f'could not open <{url}>')
			return nx.Graph()
		# r = requests.get(url)
		soup = BeautifulSoup(r.content,'html.parser')
		s = soup.find('div', class_="site-body")
		movielist = s.find('ul', class_="poster-list -p70 -grid film-list clear")
		content = movielist.find_all('li')
		for line in content:
			img = line.find('img', class_='image')
			name = img['alt']
			persons_movies.append(name)
	nodes = []
	not_in = []
	for mov in persons_movies:
		if mov in title_ID_dict.keys():
			nodes += title_ID_dict[mov]
		else:
			not_in.append(mov)

	## not_in will be [recent-movies / shortfilms / tvshows] on letterboxd but not in current graph 
	# print("The following movies are not in the graph ==> ", not_in) 

	## subgraph of your movies
	nodes = [int(n) for n in nodes]
	return G.subgraph(nodes)

def print_close(personal):
	cl_cent = sorted(nx.closeness_centrality(personal).items(), reverse=True, key=lambda x : x[1])
	print("\tCloseness Centrality:")
	just = max(cl_cent[:10], key = lambda x : len(str(ID_title_dict[str(x[0])])))
	just = len(str(ID_title_dict[str(just[0])]))
	for i in range(min(len(cl_cent),10)):
		print('{} : {:.4f}'.format(str(ID_title_dict[str(cl_cent[i][0])]).ljust(just),cl_cent[i][1]))

def print_deg(personal):
	deg_cent = sorted(nx.degree_centrality(personal).items(), reverse=True, key=lambda x : x[1])
	print("\tDegree Centrality:")
	just = max(deg_cent[:10], key = lambda x : len(str(ID_title_dict[str(x[0])])))
	just = len(str(ID_title_dict[str(just[0])]))
	for i in range(min(len(deg_cent),10)):
		print('{} : {:.4f}'.format(str(ID_title_dict[str(deg_cent[i][0])]).ljust(just),deg_cent[i][1]))

def genre_bd(movieIDs, ID_title_dict):
	genes = dict()
	num_genres = 0
	for ID in movieIDs:
		mov = ID_title_dict[str(ID)]
		for g in mov.genres:
			num_genres += 1
			if g in genes:
				genes[g] += 1
			else:
				genes[g] = 1
	return sorted(list(genes.items()), reverse = True, key = lambda x: x[1]), num_genres

def print_genre_bd(movieIDs,ID_title_dict):
	genes, num_genres = genre_bd(movieIDs, ID_title_dict)
	print('\tGenres:')
	for i in range(min(len(genes),10)):
		print('{} : {} : {:.2f}%'.format(genes[i][0].ljust(20), str(genes[i][1]).ljust(5), 100*genes[i][1]/num_genres))

def year_bd(movieIDs, ID_title_dict, decades):
	yrs = {}
	for ID in movieIDs:
		mov = ID_title_dict[str(ID)]
		yr = mov.year
		yr -= yr%10 if decades else 0
		if yr in yrs:
			yrs[yr] += 1
		else: 
			yrs[yr] = 1
	return sorted(list(yrs.items()), reverse =True, key = lambda x:x[1])

def print_year_bd(movieIDs, ID_title_dict, decades = False):
	years = year_bd(movieIDs, ID_title_dict, decades)
	print(f"Most-Watched {'decades' if decades else 'years'} for movies:")
	for i in range(min(10,len(years))):
		print('{}{} : {} : {:.2f}%'.format(years[i][0], 's' if decades else '', str(years[i][1]).ljust(3), 100*years[i][1]/len(movieIDs)))

def director_bd(movieIDs, ID_title_dict):
	dirs = {}
	for ID in movieIDs:
		mov = ID_title_dict[str(ID)]
		director = mov.director
		if director in dirs:
			dirs[director] += 1
		else: 
			dirs[director] = 1
	return sorted(list(dirs.items()), reverse =True, key = lambda x:x[1])

def print_director_bd(movieIDs, ID_title_dict):
	dirs = director_bd(movieIDs, ID_title_dict)
	adjust = max(dirs[:10], key = lambda x: len(x[0]))
	adjust = len(adjust[0])
	print(f"\tMost-Watched Directors:")
	for i in range(min(10,len(dirs))):
		print('{} : {} : {:.2f}%'.format(dirs[i][0].ljust(adjust), str(dirs[i][1]).ljust(3), 100*dirs[i][1]/len(movieIDs)))

def draw_genres(g, genre_groups):
	colors = []
	for nodeID in g.nodes():
		mov  = ID_title_dict[str(nodeID)]
		genres = mov.genres
		first_g = 'none'
		if len(genres) > 0:
			first_g = genres[0] # how could you account for the multiple genres of a single movie??
		colors.append(genre_groups[first_g])
	if g.order() < 200:
		nx.draw_kamada_kawai(g, node_size = 30, node_color=colors)
	else:
		nx.draw(g, node_size = 15, node_color=colors)
	plt.show()

def deg_label_prop_printer(BCC, ID_title_dict, draw = False, g_bd = False):
	Cl = label_prop(BCC)
	print('Edge cut   : ', edge_cut(BCC,Cl))
	print('Conductance: ', conductance(BCC,Cl))
	print('Modularity : ', modularity(BCC, Cl))
	if draw:
		draw_comm_graph(BCC, Cl)
	unique_comms = list(set([x for x in Cl.values()]))
	print('# comms:', len(unique_comms))
	comm_nodes = []
	for c in unique_comms:
		tmp = []
		for n in BCC.nodes():
			if Cl[n] == c:
				tmp.append(n)
		comm_nodes.append(set(tmp))
	comm_nodes = sorted(comm_nodes, key = lambda x: len(x), reverse = True)
	print('size of largest community: ', len(comm_nodes[0]))
	for i in range(1,len(comm_nodes)):
		print("community:")
		for c in comm_nodes[i]:
			print(ID_title_dict[str(c)], end = "  ")
		print('\n')
		if g_bd:
			print_genre_bd(comm_nodes[i], ID_title_dict)
	return Cl, comm_nodes

def get_director_comms(g, ID_title_dict):
	dir_comms, dir_labels = {}, {}
	for ID in g.nodes():
		mov = ID_title_dict[str(ID)]
		if mov.director not in dir_labels:
			dir_labels[mov.director] = ID
			dir_comms[ID] = ID
		else:
			dir_comms[ID] = dir_labels[mov.director]
	return dir_comms

## find most well connected verts NOT in subgraph ==> movie recs based on what you've seen
def get_recs(personal, G, ID_title_dict, rec_type = 'degree'):
	neighbs = set()
	p2 = personal.copy()
	pers_nodes = set(personal.nodes())

	for node in personal.nodes():
		neighbs.update(set(G.neighbors(node)))
	neighbs = neighbs - pers_nodes

	for n in neighbs: # MUCH faster
		edges_to_add = set(G.neighbors(n)) & pers_nodes
		for e in edges_to_add:
			p2.add_edge(n,e)
			p2[n][e].update(G.get_edge_data(n,e)) # keep same weight from G
	
	####################################
	## Old way => make subgraph of all neighbs and personal, remove edges between nodes in neighbs
	## Problem => WAY more edges to remove than are added, slower than a snail on lazy sunday
	# recs_graph = G.subgraph(neighbs | set(personal.nodes())).copy()
	# nonos = G.subgraph(neighbs).edges()	
	# for no in nonos:
	# 	recs_graph.remove_edge(*no)
	#####################################

	if rec_type == 'degree': 
		## sort by number of movies that share (a least 1) common actor
		bydegree = sorted(list(p2.degree(neighbs)), reverse = True, key = lambda x: x[1])
		return bydegree
	elif rec_type == 'closeness':
		## sort by number of total actors shared with movies you've watched
		cl = []
		for n in neighbs:
			shared_actors = 0
			for gs in p2.neighbors(n):
				shared_actors += p2[n][gs]["weight"]
			cl.append((n,1/shared_actors))

		closer = sorted(cl, key = lambda y: y[1])
		return closer



#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################


# Make sure you have initiated + pickled graphs from 'project.py' and 'weighted_graph.py'
with open('save_vars.pickle', 'rb') as f:
	_,ID_title_dict = pickle.load(f)

title_ID_dict, genre_groups, c = dict(), {'none':0}, 1

for ID,mov in ID_title_dict.items():
	for g in mov.genres:
		if g not in genre_groups:
			genre_groups[g] = c
			c += 10
	tit = mov.title
	if tit not in title_ID_dict.keys():
		title_ID_dict[tit] = [ID]
	else:
		title_ID_dict[tit].append(ID)

with open('save_vars_weighted.pickle', 'rb') as f:
	G = pickle.load(f)

#########################################################################################################

# user = 'lunasabine' # smaller graph (74 verts)
user = 'dolypoly' # my account
# user = 'kurstboy' # larger graph (~1000 verts) 

personal = scrapey(user, G, title_ID_dict)
# nx.write_graphml(personal, f'{user}.graphml')

CCs = sorted(list(nx.connected_components(personal)), reverse = True, key = lambda x:len(x))
largest_CC = CCs[0]
BCC = G.subgraph(largest_CC)

print("Number of movies: ", personal.order())
print("Number of edges : ", personal.size())
print(f'Number of connected components: {len(CCs)}')
print(f'Num trivial components: {len([i for i in CCs if len(i) == 1])}')
print("Size of largest connected comp: ", len(largest_CC))


#########################################################################################################

## create coloring based on genre (kinda dumb bc movie can have > 1 genre)
# draw_genres(personal, genre_groups)

## assess centralities (closeness is hard if lots edges and weight accessing)
# print_close(personal)
# print_deg(personal)

#########################################################################################################

## find breakdown of genres/years
# print_genre_bd(personal.nodes(),ID_title_dict)
# print_year_bd(personal.nodes(),ID_title_dict, decades= False)
# print_year_bd(personal.nodes(),ID_title_dict, decades= True) 
# print_director_bd(personal.nodes(), ID_title_dict)

#########################################################################################################

## find communities of largest connected component

Cl, comm_nodes_deg = deg_label_prop_printer(BCC, ID_title_dict)

# Cw = label_prop_weighted(BCC)
# print('Weighted Label Prop:')
# print('Edge cut    : ', edge_cut(BCC,Cw))
# print('Conductance : ', conductance(BCC,Cw))
# print('Modularity  : ', modularity(BCC, Cw))

# unique_comms, comm_nodes = list(set([x for x in Cw.values()])), []
# print("# communities: ", len(unique_comms))
# for c in unique_comms:
# 	tmp = []
# 	for n in BCC.nodes():
# 		if Cw[n] == c:
# 			tmp.append(n)
# 	comm_nodes.append(set(tmp))
# comm_nodes = sorted(comm_nodes, key = lambda x: len(x), reverse = True)
# print('size of largest community: ', len(comm_nodes[0]))
# for i in range(1,len(comm_nodes)):
# 	print(f"community of size {len(comm_nodes[i])}:")
# 	for c in comm_nodes[i]:
# 		print(ID_title_dict[str(c)], end = "  ")
# 	print('\n')
# 	# print_genre_bd(comm_nodes[i], ID_title_dict)

# draw_comm_graph(BCC, Cw)
# draw_comm_graph(BCC, Cl)


#########################################################################################################

## Risk analysis
# risky = sum([1 for cc in CCs if len(cc)==1])
# print("{:.2f}% of movies you watch do not have any shared, familiar actors".format(100*risky/personal.order()))

#########################################################################################################

## recommendation based on familiar actors
# deg_recs = get_recs(personal,G,ID_title_dict)
# print("\tDegree Centrality Recommendations:")
# max_just = 35
# adjust = max(deg_recs[:10], key = lambda x : len(str(ID_title_dict[str(x[0])])))
# adjust = len(str(ID_title_dict[str(adjust[0])])) + 7
# for i in range(10):
# 	mov = ID_title_dict[str(deg_recs[i][0])]
# 	tit_yr = str(mov) + " (" + str(mov.year) + ")"
# 	print('{} : {}'.format(tit_yr.ljust(adjust),deg_recs[i][1]))


# close_recs = get_recs(personal,G,ID_title_dict,rec_type='closeness')
# print("\tCloseness Centrality Recommendations:")
# adjust = max(deg_recs[:10], key = lambda x : len(str(ID_title_dict[str(x[0])])))
# adjust = len(str(ID_title_dict[str(adjust[0])])) + 7
# for i in range(10):
# 	mov = ID_title_dict[str(close_recs[i][0])]
# 	tit_yr = str(mov) + " (" + str(mov.year) + ")"
# 	print('{} : {:.4f}'.format(tit_yr.ljust(adjust),close_recs[i][1]))

#########################################################################################################

## Franchize detection
# heavy_edges = set()
# tol = 6
# for e in personal.edges():
# 	if personal.get_edge_data(*e)['weight'] > tol:
# 		heavy_edges = heavy_edges | {e}
# heaviest = G.edge_subgraph(heavy_edges)

# heavy_CCs = sorted(list(nx.connected_components(heaviest)), reverse = True, key = lambda x:len(x))
# heavy_largest_CC = heavy_CCs[0]

# print("Number of movies: ", heaviest.order())
# print("Number of edges : ", heaviest.size())
# print(f'Number of connected components: {len(heavy_CCs)}')
# print("Size of largest connected comp: ", len(heavy_largest_CC))

# i,draw_comms = 1,False
# for cc in heavy_CCs:
# 	print(f"{i}:", end = ' ')
# 	for ID in cc:
# 		print(ID_title_dict[str(ID)], end=" | ")
# 	print()
# 	if len(cc) > 4 and draw_comms:
# 		cc_graph = heaviest.subgraph(cc)
# 		comms = get_director_comms(cc_graph,ID_title_dict)
# 		colors = [comms[v] for v in cc_graph.nodes()]
# 		nx.draw(cc_graph, node_size = 30, node_color=colors)
# 		plt.title(i)
# 		plt.show()
# 	i+=1

# draw_comms = True
# if draw_comms:
# 	comms = get_director_comms(heaviest,ID_title_dict)
# 	colors = [comms[v] for v in heaviest.nodes()]
# 	nx.draw(heaviest, node_size = 30, node_color=colors)
# 	plt.show()

#########################################################################################################

# Compare u and ur friends tastes.
# user1 = 'dolypoly'
# user2 = 'lunasabine'
# Gu1 = scrapey(user1, G, title_ID_dict)
# Gu2 = scrapey(user2, G, title_ID_dict)

# print(f"Number of movies you've both seen: {len(set(Gu1.nodes()) & set(Gu2.nodes()))}")

# risky_1 = sum([1 for cc in nx.connected_components(Gu1) if len(cc)==1])
# risky_2 = sum([1 for cc in nx.connected_components(Gu2) if len(cc)==1])
# abs_diff = abs(risky_1-risky_2)
# if abs_diff > 0:
# 	print(f'{user1 if risky_1 > risky_2 else user2} takes more risks, watching {abs_diff} more movies with no actors in any other watched movies')
# 	print(f'{user1} : {risky_1}      {user2} : {risky_2}')
# else:
# 	print(f'{user1} and {user2} watch the same amount of risky movies ({risky_1})')

