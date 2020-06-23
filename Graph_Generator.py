#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
from sklearn.cluster import KMeans
from pyclustering.cluster import kmeans , clique
from sklearn.mixture import BayesianGaussianMixture
import os
from itertools import combinations 
from itertools import product
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import fowlkes_mallows_score
import sklearn
import warnings
warnings.filterwarnings("ignore")

#parameters
nNodes=70
nClique=4
maxCliqueSize=30
density=0.5


####### enter number of clique along with number of respective edges manually below ################
size_of_clique=[]
no_of_clique=nClique
nodeSum=0
residue= (int)((nNodes-maxCliqueSize)/(nClique-1))

nodeSum=maxCliqueSize
for i in range(no_of_clique-1):
    x=np.random.randint(3,residue-1)
    size_of_clique.append(x)

size_of_clique.append(maxCliqueSize)

print(size_of_clique)
total_no_of_nodes=nNodes
print("Total number of nodes in the graph ", total_no_of_nodes)

node_id=list(range(1,total_no_of_nodes+1))

print("Node ids")
print(node_id)

clique_list = [[] for j in range(no_of_clique)]

size=0
clique_nodes=[]

for i in range(no_of_clique):
    
    clique_list[i]=list(range((size+1),(size+size_of_clique[i]+1)))
    x=clique_list[i]
    for k in range(len(clique_list[i])):
        clique_nodes.append(x[k])
    size=size+size_of_clique[i]
    
print("Cliques")    
print(clique_list) 


######### save clique lists ###################

with open("ground_truth_clique_list.txt", "w") as gtcl:
    for index5 in range(len(clique_list)):
        cluster_set=clique_list[index5]
        cluster_set_size=len(cluster_set)
        for index6 in range(cluster_set_size):
            gtcl.write("%s " % int(cluster_set[index6]))
        gtcl.write("\n")
        

######## find all comb. inter clique / non clique edges  ###############

non_clique_all_comb=[]

for i in range(no_of_clique):
    for j in range(i+1,no_of_clique):
        
        clique_set1=clique_list[i]
        clique_set2=clique_list[j]
        
        new_comb=(list(product(clique_set1,clique_set2)))
        #print(new_comb)
        non_clique_all_comb.extend(new_comb)
    

#edges from non-clique-vertices


edges=[]
for i in range(len(node_id)):
    if i not in clique_nodes:
        for j in range(i+1, len(node_id)):
            x=[]
            x.append(i)
            x.append(j)
            edges.append(x)

for i in range(len(non_clique_all_comb)):
    x=non_clique_all_comb[i]
    t=[]
    t.append(x[0])
    t.append(x[1])
    edges.append(t)


non_clique_edges=edges

########### sample inter clique / non clique  edges ###################
random_selected_non_clique_edges=[]
nPortion=(int)(len(edges)*density)
idx = np.random.randint(len(non_clique_edges), size=nPortion)
for i in range(len(idx)):  
    random_selected_non_clique_edges.append(non_clique_edges[idx[i]])


######### save edge lists ###################
total_clique_edges=0
with open("edge_file.txt", "w") as edgeid:
    
    for i in range(no_of_clique):
        clique_set=clique_list[i]
        
        clique_all_comb=np.array(list(combinations(clique_set,2)))
        total_clique_edges= total_clique_edges+len(clique_all_comb)
        total_clique_edges= total_clique_edges+len(clique_all_comb)
        
        for i1 in range(len(clique_all_comb)):
            
            first_comb_clique=clique_all_comb[i1]
            
            n_source_node=first_comb_clique[0]
            n_target_node=first_comb_clique[1]
            
            
            edgeid.write("%s " % int(n_source_node))
            edgeid.write("%s " % int(n_target_node))
            edgeid.write("\n") 

    for i in range(len(random_selected_non_clique_edges)):

        first_comb_non_clique=random_selected_non_clique_edges[i]

        source_node=first_comb_non_clique[0]
        target_node=first_comb_non_clique[1]

        edgeid.write("%s " % int(source_node))
        edgeid.write("%s " % int(target_node))
        edgeid.write("\n")
edgeid.close()

print("Number of clique edges ",total_clique_edges)
print("Number of non-clique edges ",len(non_clique_edges))
print("Number of randomly selected non-clique edges ", len(random_selected_non_clique_edges))


# In[ ]:





# In[ ]:




