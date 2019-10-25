# -*- coding: utf-8 -*-
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from numpy import linalg as la

def cluster(MIN_K,MAX_K,Samples):
    objFunctions = []
    
    for i in range(MIN_K,MAX_K+1):
        print("Working on k="+str(i))
        numberOfSamples = len(Samples)
        numOfTuples = len(Samples[0])
        centroids = np.ndarray(shape=(i,numOfTuples),dtype=float)
        
        prev = np.ndarray(shape=(i,numOfTuples))
        clusters = dict()
        
        #Selecting random points
        for j in range(i):
            centroids[j] = Samples[rd.randint(0,numberOfSamples-1)]
        
        while not np.array_equal(centroids,prev):
            clusters = dict()
            euclideanNorms = np.asarray([la.norm(Samples- k,axis=1) for k in centroids]).T
            indices = np.argmin(euclideanNorms,axis=1)
            prev = centroids.copy()
            for m in range(len(indices)):
                if not indices[m] in clusters:
                    clusters[indices[m]] = []
                clusters[indices[m]].append(Samples[m])
            
            centroids = np.asarray([np.mean(np.asarray(clusters[p]),axis=0) for p in sorted(list(clusters.keys()))])
        
        #Plotting the clusters
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('Plot for K='+str(i))
        colors = ['#ff00ff','#6666ff','#00ccff','#00ff99','#009900','#cccc00','#ff9933','#669999','#b35900','#0000ff']
        objectiveFunctionValue = 0          
        for t in sorted(list(clusters.keys())):
            clusterPoints = np.asarray(clusters[t])
            transposeMatrix = clusterPoints.T
            ax.scatter(transposeMatrix[0],transposeMatrix[1],c=colors[t],s=20)
            ax.scatter(centroids[t][0],centroids[t][1],c='#000000',marker="x",s=50)
            distance = clusterPoints - centroids[t]
            euclideanDistance = la.norm(distance, axis=1)
            squaredDistance = np.square(euclideanDistance)
            objectiveFunctionValue += np.sum(squaredDistance)
        objFunctions.append(objectiveFunctionValue)

    objPlotFig = plt.figure()
    plotFigure = objPlotFig.add_subplot(1,1,1)
    plotFigure.plot([h for h in range(MIN_K,MAX_K+1)],objFunctions)
    plotFigure.set_xlabel('K')
    plotFigure.set_ylabel('Objective Function')
    plotFigure.set_title('Objective Function vs K')
    
    return objFunctions
