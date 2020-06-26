#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from pyclustering.cluster import kmeans , clique

from sklearn.mixture import BayesianGaussianMixture
import os

from itertools import combinations 


from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import fowlkes_mallows_score

import warnings
warnings.filterwarnings("ignore")

import sklearn
sklearn.__version__

nCluster=6
nNodes=70
maxClique=30
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
        print("For cluster number ",j, "fraction of total Cluster size", (len(l)/len(c)), " fraction of clique ", (len(l)/len(clique)))
        if maxCount < len(l):
            maxCount=len(l)
            index=j
##Returning top 2 clusters
            
def findMaxCluster(clusters, maxCliqueSize):
    maxSize=0
    maxCluster=[]
    for i in range(len(clusters)):
        cluster=clusters[i]
        if maxSize < len(cluster):
            maxSize=len(cluster)
            maxCluster=cluster
    print("Size of maximum cluster ", maxSize)
    secCluster=[]
    secMax=0
    for i in range(len(clusters)):
        cluster=clusters[i]
        if maxSize != len(cluster):
            if secMax < len(cluster):
                secMax=len(cluster)
                secCluster=cluster
    print("Size of second maximum cluster ", secMax)
    print("Fraction of joint cluster size with respect to total number of nodes", (maxSize+secMax)/nNodes)
    for i in range(len(secCluster)):
        maxCluster.append(secCluster[i])
    return maxCluster
    
def evaluateprediction(cluster, clique):
    count=0
    for i in range(len(clique)):
        if clique[i] in cluster:
            count = count+1
    print(count, len(cluster))
    print("Precision ", count/len(cluster))
    print("Recall ", count/len(clique))

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

x=findMaxCluster(clusters,maxClique)
print("MaxClique")
#print(x)
evaluateprediction(x, clique)


# In[ ]:




