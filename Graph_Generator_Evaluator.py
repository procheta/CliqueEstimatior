#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import networkx as nx

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

####### enter number of clique along with number of respective edges manually below ################

no_of_clique=4
size_of_clique=[30,20,10,5]

total_no_of_nodes=np.sum(size_of_clique)
print("Total number of nodes in the graph ", total_no_of_nodes)

node_id=list(range(1,total_no_of_nodes+1))

print("Node ids")
print(node_id)

clique_list = [[] for j in range(no_of_clique)]

size=0

for i in range(no_of_clique):
    
    clique_list[i]=list(range((size+1),(size+size_of_clique[i]+1)))
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
    for j in range(i):
        
        clique_set1=clique_list[i]
        clique_set2=clique_list[j]
        
        new_comb=(list(product(clique_set1,clique_set2)))
        
        non_clique_all_comb.extend(new_comb)
    


non_clique_edges=np.array(non_clique_all_comb)




########### sample inter clique / non clique  edges ###################


idx = np.random.randint(len(non_clique_edges), size=20)

random_selected_non_clique_edges=np.array(non_clique_edges[idx,:])


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
######## generate and plot graph ######################

G1=nx.read_edgelist("edge_file.txt")
edge_list=list(G1.edges())
node_id1=list(G1.nodes())

#pos=nx.spring_layout(G1) # positions for all nodes
#nx.draw_networkx(G1,pos,nodelist=node_id1, node_color='r', node_size=200, alpha=0.8)


#Evaluation of Random Walk

nCluster=6
#function to find similarity between two vecs
def findSimilarity(vec1,vec2):
    sim=0
    for i in range(len(vec1)):
        sim = sim+vec1[i]*vec2[i]
    return sim

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def findClusterOverlap(clusters,clique):
    maxCount=0
    index=0
    for j in range(len(clusters)):
        c=clusters[j]
        clusterSize=len(c)
        count=0
        l=intersection(c,clique)
        print("Cluster number ",j, " overlap ", len(l), "fraction of total Cluster size", (len(l)/len(c))*100, "% fraction of clique ", (len(l)/len(clique)))
        if maxCount < len(l):
            maxCount=len(l)
            index=j
        
WordVecDic={}
vecs=[]
ids=[]
flag=0
with open("output", "r") as f:
    for line in f:
        if flag == 0:
            flag=1
        else:
            vec=[]
            x=line.split("\n")[0]
            words=x.split(" ")
            key=words[0]
            count=0
            for word in words:
                if count>=1:
                    vec.append(float(word))
                else:
                    count=1
            vecs.append(vec)
            ids.append(int(key))
            WordVecDic[key]=vec
                        
clique_list=[]
maxCliqueSize=0
maxCliqueIndex=0
index=0

with open("ground_truth_clique_list.txt", "r") as gtcl:
    for line in gtcl:
        x=line.strip().split(" ")
        clique=[]
        for x1 in x:
            clique.append(int(x1))
        clique_list.append(clique)
        if maxCliqueSize < len(clique):
            maxCliqueSize=len(clique)
            maxCliqueIndex=index
        index=index+1
            
        
print("max clique index ",maxCliqueIndex) 
print("size of maximum clique ",maxCliqueSize)
maxClique=clique_list[maxCliqueIndex]

###computing similarity within cluster######
avgSim=0
for i in range(len(maxClique)):
    vec1=WordVecDic[str(maxClique[i])]
    for j in range(i+1,len(maxClique)):
        vec2= WordVecDic[str(maxClique[j])]
        sim=findSimilarity(vec1,vec2)
        avgSim=avgSim+sim

avgSim=avgSim/(len(maxClique)*(len(maxClique)-1)/2)

print("Average similarity within cluster", avgSim)

otherNodesSim=0

for i in range(len(maxClique)):
    vec1=WordVecDic[str(maxClique[i])]
    for j in range(len(ids)):
        if ids[j] not in maxClique:
            vec2= WordVecDic[str(ids[j])]
            sim=findSimilarity(vec1,vec2)
            otherNodesSim=otherNodesSim+sim
        

otherNodesSim = otherNodesSim/((len(ids)-len(maxClique))*len(maxClique))
print("Average similarity of clique with other nodes", otherNodesSim)

print("ratio of within clique similarity vs other node similarity ", avgSim/otherNodesSim)


kmeans = KMeans(n_clusters=nCluster, random_state=0).fit(vecs)

clusters=[]
for i in range(nCluster):
    x=[]
    clusters.append(x)
    
for i in range(len(kmeans.labels_)):
    clusters[kmeans.labels_[i]].append(int(ids[i]))

print("Custering completed")    


clique=clique_list[maxCliqueIndex]
findClusterOverlap(clusters,clique)


# In[ ]:





# In[ ]:




