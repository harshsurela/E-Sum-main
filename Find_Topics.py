# import commands
import string
import os
import math
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain


## Removes Punctuation from text
def strip_punctuation(s):
        table = str.maketrans({key: None for key in string.punctuation})
        return s.translate(table)


def coocurrence_frequency(A,B):
        count = 0
        for i in range(A.shape[0]):
                        if(A[i]*B[i]!=0):
                                count+=1
        return(count)

def Log2(x):
        if(x<=0):
                return 0
        else:
                return math.log(x,2)
        
## Function for NMF decomposition of term-sentence matrix A
def tfisf(f_name):
        
        ## Reads term-sentence frequency matrix from .csv file and stores in matrix A
        termSentFile = ".\\Pre_Processed\\"+f_name.replace('.txt','')+".csv"
        data = np.genfromtxt(termSentFile, dtype='float64', delimiter=',', names=None)
        A = np.asarray(data,dtype = 'float64')
                        
        TT_matrix = np.matmul(A,np.transpose(A))

        G = nx.Graph()

        for i in range(TT_matrix.shape[0]):
                for j in range(TT_matrix.shape[0]):
                        if(i>j and  TT_matrix[i][j]>1):
                                w = TT_matrix[i][j]
                                G.add_edge(i,j,weight=w)
                                
       
        # compute the best partition
        partition = community_louvain.best_partition(G,randomize=False)
        clusters = len(set(list(partition.values())))
        elements_in_clusters = np.zeros(clusters, dtype= 'int32')

        for i in range(clusters):
                res = 0
                for key in partition:  
                    if partition[key] == i:  
                        res = res + 1
                elements_in_clusters[i]=res     
        
        count = 0
        for i in range(clusters):
                if(elements_in_clusters[i]>3):
                        count+=1
       
        topics.append(count)

            
                        
os.chdir("H:\WORKSPACE\Python\E-Summ-main")
topics = []
for f in os.listdir(".\\Documents"):
    inputFile = ".\\Documents\\"+f
    tfisf(f)
file_obj = ".\\Sample Doc.csv"
np.savetxt(file_obj, np.array(topics).transpose(), fmt='%1.5f', delimiter=",")




