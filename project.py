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

PICKLEIT = 1

'''
NOT INCLUDED IN SUBMISSION: DATASETS
visit <https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset> 
to download the "credits.csv" & "movies_metadata.csv" datasets used
'''
def initialize():
       credits_df = pd.read_csv('credits.csv') ## DEFINE DATATYPES
       crew = credits_df.drop(['cast'],axis=1)
       credits_df.drop(['crew'],axis=1,inplace=True)

       ID_dir_dict = {}
       for idx, row in crew.iterrows():
              whole_crew = ast.literal_eval(row.crew)
              director = ''
              for member in whole_crew:
                     #member == dict
                     if member['job'] == 'Director':
                            director = member['name']
              ID_dir_dict[str(row.id)] = director

       # now have dataframe of MOVIEs and corresp list of ACTORs
       # read csv movie_metadata to assign titles to movie id's, and create nodes for movie,
       # then attach initialized or existing actor nodes to movie. 

       movie_df = pd.read_csv('movies_metadata.csv',  dtype={'adult':str, 'belongs_to_collection':str, 'budget':str, 'genres':str, 'homepage':str, 'id':str,
              'imdb_id':str, 'original_language':str, 'original_title':str, 'overview':str,
              'popularity':str, 'poster_path':str, 'production_companies':str,
              'production_countries':str, 'release_date':str, 'revenue':str, 'runtime':str,
              'spoken_languages':str, 'status':str, 'tagline':str, 'title':str, 'video':str,
              'vote_average':str, 'vote_count':str})

       # drop all but ['budget', 'genres', 'id', 'original_title', 'release_date', 'title'] columns (future meta-data)
       movie_df.drop(['adult', 'belongs_to_collection', 'homepage',
              'imdb_id', 'original_language', 'overview',
              'popularity', 'poster_path', 'production_companies',
              'production_countries', 'revenue', 'runtime',
              'spoken_languages', 'status', 'tagline', 'video',
              'vote_average', 'vote_count'], axis = 1, inplace=True)
       movie_df.drop(['budget', 'original_title'], axis = 1, inplace=True) 
       # now only have df with ['id','title', 'genres', 'release_date']


       movie_IDs = movie_df['id'].tolist()
       movie_titles = movie_df['title'].tolist()
       movie_years = movie_df['release_date'].tolist()
       movie_genres = movie_df['genres'].tolist()
       movie_genres = [list(ast.literal_eval(x)) for x in movie_genres]
       act_gnrs = []
       for genes in movie_genres:
              tmp = []
              for g in genes:
                     tmp.append(g['name'])
              act_gnrs.append(tmp)
       movie_genres = act_gnrs
       # print(act_gnrs[:5])
       int_yrs = []
       for i in range(len(movie_years)):
              yr = str(movie_years[i])[:4]
              intyr = 0
              try:
                     intyr = int(yr)
              except:
                     intyr = 0
              int_yrs.append(intyr)
       movies_obj_list = []
       for i in range(len(movie_IDs)):
              mov_id = movie_IDs[i]
              try:
                     director = ID_dir_dict[mov_id]
              except:
                     director = ''
              movies_obj_list.append(Movie(mov_id, movie_titles[i], int_yrs[i], movie_genres[i], director))
       ID_title_dict = dict(zip(movie_IDs,movies_obj_list))

       g = nx.Graph()

       g.add_nodes_from(credits_df.id, bipartite=0)
       # predicted_edge_num = 0

       for idx, row in tqdm(credits_df.iterrows(),total=credits_df.shape[0]):
              mov = row.id
              tmp_actors = ast.literal_eval(row.cast)
              # predicted_edge_num += len(tmp_actors)
              for actor in tmp_actors:
                     g.add_node(actor['name'],bipartite=1)
                     g.add_edge(actor['name'],mov)

       with open('save_vars.pickle', 'wb') as f:
                  f.write(pickle.dumps((g,ID_title_dict)))
       
       return g,ID_title_dict


#########################################################################################################
#########################################################################################################


# nodes = movie numbers, actor names, metadata=title
# for movie in credits.csv, add movie, add actor if not already in, add edge between two.
# use movies_metadata.csv to add title for each movie number
if not PICKLEIT:
       g,ID_title_dict = initialize()
else:
       with open('save_vars.pickle', 'rb') as f:
                  g,ID_title_dict = pickle.load(f)


actors = {n for n, d in g.nodes(data=True) if d['bipartite'] == 1}
movies = set(g) - actors

print(g.order())
print(g.size())
# nx.write_graphml(g,'bipart.graphml')

#number of trivial actors (one movie only)
num_triv,max_deg = 0,0
max_deg_actor = 'Dolan Clahan'
actor_degrees = {}
for actor in actors:
       deg = g.degree(actor)
       if deg not in actor_degrees:
              actor_degrees[deg] = 1
       else:
              actor_degrees[deg] += 1
       num_triv += 1 if deg == 1 else 0
       max_deg_actor = actor if deg > max_deg else max_deg_actor
       max_deg = max(deg,max_deg)


print(f'Number of trivial actors:  {num_triv}')
print(f'Actor in most movies: {max_deg_actor}, in {max_deg} movies')


movie_degrees = {}
zero_deg_mov = []
max_deg_movie = ''
num_zero_m, num_triv_m, max_deg_m = 0,0,0
for m in movies:
       d = g.degree(m)
       if d == 0:
              num_zero_m += 1
              zero_deg_mov.append(m)
              continue
       num_triv_m += 1 if d == 1 else 0
       max_deg_movie = m if d > max_deg_m else max_deg_movie
       max_deg_m = max(max_deg_m,d)
       if d not in movie_degrees:
              movie_degrees[d] = 1
       else:
              movie_degrees[d] += 1

g.remove_nodes_from(zero_deg_mov) #dont need any movies w no actors
movies = movies - set(zero_deg_mov)
print(f'Number of movies with zero actors:  {num_zero_m}; new # movies = {len(movies)}')
print(f'Number of trivial movies (1 actor): {num_triv_m}')
print(f'Movie with most actors: {ID_title_dict[str(max_deg_movie)]}, with {max_deg_m} actors')

print(ID_title_dict[str(max_deg_movie)].info())

sorted_actor_degrees = sorted(actor_degrees.items())
sorted_movie_degrees = sorted(movie_degrees.items())
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([k for (k,v) in sorted_actor_degrees], [v for (k, v) in sorted_actor_degrees], 'b')
ax.plot([k for (k,v) in sorted_movie_degrees], [v for (k, v) in sorted_movie_degrees], 'r')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('degree')
ax.set_ylabel('frequency')
ax.set_title('degree distribution of actors(blue) and movies(red)')
plt.show()

#####################################################################################################################

CCs = list(nx.connected_components(g))
print(f'Number of connected components: {len(CCs)}')
#get size distribution for connected components
sortedCCs = sorted(CCs, reverse=True, key=lambda x:len(x))
largest_CC = sortedCCs[0]
largCCactors = set([act for act in largest_CC if act in actors])
largCCmovies = set([mov for mov in largest_CC if mov in movies])
print(f"Len of Largest Connected Component:     {len(largest_CC)},\n\t\t with {len(largCCactors)} actors and {len(largCCmovies)} movies")
# print(f"2nd Largest connected component: {len(sortedCCs[1])}")
# print([len(sortedCCs[i]) for i in range(50)])

#########################################################################################################

#             degree skew
def pl_ex_coef(degrees, order):
       xmin = min(degrees.keys())
       ahat = 1  +  (order)  *  (1/sum([v*math.log(k/xmin) for (k,v) in degrees.items()]))
       pl_coef = (ahat-1)/xmin
       return ahat, pl_coef

actor_ahat, actor_pl_coefficient = pl_ex_coef(actor_degrees,len(actors))
movie_ahat, movie_pl_coefficient = pl_ex_coef(movie_degrees,len(movies))

print("\nActor Power-law exponent:", actor_ahat)
print("Actor Power-law coefficient:", actor_pl_coefficient)
print("Movie Power-law exponent:", movie_ahat)
print("Movie Power-law coefficient:", movie_pl_coefficient)

#########################################################################################################

# degree distribution of whole graph
total_degrees = {} 
for v in g.nodes():
       d = g.degree(v)
       if d not in total_degrees:
              total_degrees[d] = 1
       else:
              total_degrees[d] += 1
total_ahat,total_pl_coef = pl_ex_coef(total_degrees,g.order())
print("\nTotal Power-law exponent:", total_ahat)
print("Total Power-law coefficient:", total_pl_coef)


sorted_total_degrees = sorted(total_degrees.items())
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([k for (k,v) in sorted_total_degrees], [v for (k, v) in sorted_total_degrees], 'r')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('degree')
ax.set_ylabel('frequency')
ax.set_title('degree distribution for whole graph')
plt.show()

#########################################################################################################

#                    hubs
def hub_ratio(degrees,order):
       cutoff = math.log(order)
       tail = [v for (k,v) in degrees.items() if k > cutoff]
       hub_ratio = sum(tail)/order
       return hub_ratio

actor_hub_ratio = hub_ratio(actor_degrees,len(actors))
movie_hub_ratio = hub_ratio(movie_degrees,len(movies))
total_hub_ratio = hub_ratio(total_degrees,g.order())
print("\nRatio of hubs among set of actors:", actor_hub_ratio)
print("Ratio of hubs among set of movies:", movie_hub_ratio)
print("Ratio of hubs on total graph:", total_hub_ratio)
print("\n\n")

#########################################################################################################

# #             small-world estimate
CC = g.subgraph(largest_CC)
k,cutoff,sum_short_path,sum_of_nodes = 100,50,0,0
ACT_sample_verts = random.choices(list(CC.nodes()),k=k)
for vert1 in ACT_sample_verts:
       sssp = nx.single_source_shortest_path_length(CC, vert1, cutoff=cutoff)
       sum_of_nodes += len(sssp.values())
       sum_short_path += sum(sssp.values())
avg_shortest_paths = sum_short_path/sum_of_nodes
print("\nAverage shortest path length actors:", round(avg_shortest_paths, 0)) # == 7
k,cutoff,sum_short_path,sum_of_nodes = 100,50,0,0
MOV_sample_verts = random.choices(list(largCCmovies),k=k)
for vert1 in MOV_sample_verts:
       sssp = nx.single_source_shortest_path_length(CC, vert1, cutoff=cutoff)
       sum_of_nodes += len(sssp.values())
       sum_short_path += sum(sssp.values())
avg_shortest_paths = sum_short_path/sum_of_nodes
print("\nAverage shortest path length movies:", round(avg_shortest_paths, 0)) # == 7

#########################################################################################################

# wait... HOW many degrees of Kevin Bacon?
length = nx.single_source_shortest_path_length(g,"Kevin Bacon")
print("Average shortest path length for Kevin Bacon:", sum(length.values())/len(length))

#########################################################################################################

#             centrality
deg_cent = sorted(nx.degree_centrality(g).items(), reverse=True, key=lambda x : x[1])
ADC = [x for x in deg_cent if x[0] in actors]
i = 0
for a in ADC:
       if a[0] == "Kevin Bacon":
              print('KB degree cent:', a[1],'  rank:',  i)
              break
       i+=1

MDC = [x for x in deg_cent if x[0] in movies]
print(f"degree centrality")
print(ADC[0:10])
print([(str(ID_title_dict[str(MDC[i][0])]), MDC[i][1]) for i in range(10)])


tik = time.time()
centrality  = nx.eigenvector_centrality(g)
centralest = sorted(centrality.items(),reverse = True, key = lambda x: x[1])
actor_cent = [x for x in centralest if x[0] in actors]
movie_cent = [x for x in centralest if x[0] in movies]
tik = time.time()-tik
print(f"eigenvector centrality (in {tik} seconds)")
print(actor_cent[0:10])
print([(str(ID_title_dict[str(movie_cent[i][0])]),movie_cent[i][1]) for i in range(10)])
print()

#########################################################################################################

#             Density
# paths = nx.single_source_shortest_path(g,"Johnny Depp", cutoff = 2)
# nodes = paths.values()
# grr = set([f for x in nodes for f in x])
# grr = g.subgraph(grr)
# pos = nx.random_layout(grr)
# grr_actors = [act for act in grr if act in actors]
# grr_movies = [mov for mov in grr if mov in movies]
# print(f"num actors: {len(grr_actors)}")
# print(f"num movies: {len(grr_movies)}")
# nx.draw_networkx_nodes(grr,pos,nodelist=grr_actors,node_color='r', node_size = 5)
# nx.draw_networkx_nodes(grr,pos,nodelist=grr_movies,node_color='g', node_size = 5)
# nx.draw_networkx_edges(grr,pos,edgelist=grr.edges())
# plt.show()

#########################################################################################################

# guy = "Willem Dafoe"
# print(f"movies {guy} has been in:")
# for ms in g.neighbors(guy):
#        print(ID_title_dict[str(ms)])

