#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import networkx as nx
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import AffinityPropagation
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

nCluster=1
nNodes=100
maxCliqueSize=44
#evaluationflag=0 if we take only the maximum size
#evaluationflag=1 if we take top most two clusters based on only size
#evaluationflag=2 if we take only the maximum density cluster
#evaluationflag=3 Merge until total size greater than clique size
evaluationFlag=3

#predictionMode=0 only degree based prediction
#predictionMode=1 only centroid similarity based prediction
#predictionMode=2 both criteria
#predictionMode=3 print all criteria based results simultaneously
predictionMode=3
#function to find similarity between two vecs
def findSimilarity(vec1,vec2):
    sim=0
    for i in range(len(vec1)):
        sim = sim+vec1[i]*vec2[i]
    return sim

degreeDic={}
for i in range(1,nNodes+1):
    degreeDic[str(i)]=0


with open("C:/Users/Procheta/Downloads/clique_graph_generator_code.tar/clique_graph_generator_code/edge_file_100_1_0.8.txt", "r") as gtcl:
    for line in gtcl:
        x=line.strip().split(" ")
        x1=x[0]
        degreeDic[x1]=degreeDic[x1]+1
        x1=x[1]
        degreeDic[x1]=degreeDic[x1]+1
        
    

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
        print("For cluster number ",j, "fraction of total Cluster size", (len(c)), " fraction of clique ", (len(l)/len(clique)))
        if maxCount < len(l):
            maxCount=len(l)
            index=j
            

            

##Returning top 2 clusters            
def findMaxCluster(clusters, maxCliqueSize,WordVecDic):
    maxCluster=[]
    if evaluationFlag==0 or evaluationFlag==1:
        maxSize=0
        for i in range(len(clusters)):
            cluster=clusters[i]
            if maxSize < len(cluster):
                maxSize=len(cluster)
                maxCluster=cluster
        print("Size of maximum cluster ", maxSize)
        print("Fraction of joint cluster size with respect to total number of nodes", (maxSize)/nNodes)
        secCluster=[]
        secMax=0
    if evaluationFlag==1:
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
    if evaluationFlag==2:
        maxSim=0
        clusterIndex=0
        for i in range(len(clusters)):
            cluster=clusters[i]
            sim=findAvgSimilarityWithinCluster(cluster, WordVecDic)
            if maxSim < sim:
                maxSim = sim
                clusterIndex=i
        maxCluster=clusters[clusterIndex]
        print("max Cluster index ", clusterIndex)
    if evaluationFlag==3:
        maxSize=0
        clusterIndex=0
        centroids=[]
        simDict={}
        for i in range(len(clusters)):
            cluster=clusters[i]
            centroids.append(findCentroid(cluster, WordVecDic))
            if maxSize < len(cluster):
                maxSize = len(cluster)
                clusterIndex=i
        print("max Cluster index ", clusterIndex)
        maxCluster=clusters[clusterIndex]
        maxClusterCentroid=centroids[clusterIndex]
        similarities=[]
        for i in range(len(clusters)):
            centroidVec=centroids[i]
            if i != clusterIndex:
                sim=findSimilarity(maxClusterCentroid,centroidVec)
                simDict[sim]=i
                similarities.append(sim)
        similarities.sort()
        for i in range(len(similarities)):
            sim=similarities[i]
            index=simDict[sim]
            cluster=clusters[index]
            if len(maxCluster) < maxCliqueSize:
                print("cluster index ", index)
                for j in range(len(cluster)):
                    maxCluster.append(cluster[j])
            else:
                break;
    return maxCluster


def findAvgSimilarityWithinCluster(cluster, WordVecDic):
    similarity=0
    for i in range(len(cluster)):
        vec1=WordVecDic[str(cluster[i])]
        for j in range(i+1, len(cluster)):
            vec2=WordVecDic[str(cluster[j])]
            similarity=similarity+findSimilarity(vec1,vec2)
    if len(cluster) > 1:
        similarity=similarity/(len(cluster)*(len(cluster)-1)/2)
    return similarity


def findCentroid(cluster, vecDic):
    centroid=[]
    for i in range(len(vecDic[str(cluster[0])])):
        centroid.append(0)
    
    for i in range(len(cluster)):
        x=cluster[i]
        vec1=vecDic[str(x)]
        for j in range(len(vec1)):
            centroid[j] = centroid[j]+vec1[j]
    clusterSize=len(cluster)
    for i in range(len(centroid)):
        centroid[i] = centroid[i]/clusterSize
    return centroid

def centroidbasedPrdiction(cluster,vecDic, maxClique):
    centroid=findCentroid(cluster, vecDic)
    simValue=[]
    simDic={}
    for i in range(len(cluster)):
        x=cluster[i]
        pointVec=vecDic[str(x)]
        sim=findSimilarity(pointVec, centroid)
        simValue.append(sim)
        simDic[sim]=x
         
        
    simValue.sort()
    count=0
    clique=[]
    for i in range(len(simValue)):
        clique.append(simDic[simValue[i]])
        if count == maxClique:
            break
        count=count +1
    return clique
    

def predictClique(cluster, degreeArray, vecDic, maxClique):
    clique=[]
    clique1=[]
    if predictionMode==0 or predictionMode==2 or predictionMode==3:
        for i in range(len(cluster)):
            if degreeArray[str(cluster[i])] >= (20-1) :
                clique.append(cluster[i])
        clique1=clique
        if len(clique) != 0:
            print("Degree based Appoach")
            evaluateprediction(clique, maxClique)
        else:
            print("Degree based prediction could't predict any clique")
    if predictionMode==2:
        cluster=clique
    if predictionMode==1 or predictionMode==2 or predictionMode==3:
        clique=centroidbasedPrdiction(cluster,vecDic, maxClique)
        if len(clique) != 0:
            print("Centroid based Appoach")
            evaluateprediction(clique, maxClique)
        else:
            print("Centroid based prediction could't predict any clique")
    if predictionMode==3:
        clique=[]
        if len(clique1) !=0:
            clique=centroidbasedPrdiction(clique1,vecDic, maxClique)
        if len(clique) != 0:
            print("Combined Appoach")
            evaluateprediction(clique, maxClique)
        else:
            print("Combined approach based prediction could't predict any clique")
    
    return clique
def evaluateprediction(cluster, clique):
    count=0
    for i in range(len(clique)):
        if clique[i] in cluster:
            count = count+1
    #print(count, len(cluster))
    print("Precision ", count/len(cluster))
    print("Recall ", count/len(clique))

WordVecDic={}
vecs=[]
ids=[]
flag=0
with open("C:/Users/Procheta/Downloads/clique_graph_generator_code.tar/clique_graph_generator_code/output_new", "r") as f:
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
            try:
                ids.append(int(key))
                WordVecDic[key]=vec
                vecs.append(vec)
            except:
                print("key ", key)
                #print(line)
                        
clique_list=[]
maxCliqueSize=0
maxCliqueIndex=0
index=0

with open("C:/Users/Procheta/Downloads/clique_graph_generator_code.tar/clique_graph_generator_code/ground_truth_clique_list_100_1_0.8.txt", "r") as gtcl:
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
print(WordVecDic.keys())
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



clustering = AffinityPropagation().fit(vecs)
cluster_centers_indices = clustering.cluster_centers_indices_
nCluster=len(cluster_centers_indices)
print("Number of clusters created", nCluster)

clusters=[]

for i in range(nCluster):
    x=[]
    clusters.append(x)
    
for i in range(len(clustering.labels_)):
    clusters[clustering.labels_[i]].append(int(ids[i]))

print("Custering completed")    


clique=clique_list[maxCliqueIndex]
findClusterOverlap(clusters,clique)

x=findMaxCluster(clusters,maxCliqueSize,WordVecDic)
print("MaxClique Predictions")
print("Max Size based prediction")
evaluateprediction(x, clique)
cl=predictClique(x, degreeDic, WordVecDic, clique)
#print("Predicted ", cl)
#print(x)
#evaluateprediction(cl, clique)


# In[ ]:




