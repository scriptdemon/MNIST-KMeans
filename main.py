import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import random as rd
from numpy import linalg as la

def getRandomCentroids(n,Samples):
    numberOfSamples = len(Samples)
    numOfTuples = len(Samples[0])
    centroids = np.ndarray(shape=(n,numOfTuples),dtype=float)
    
    for j in range(n):
        centroids[j] = Samples[rd.randint(0,numberOfSamples-1)]
    return centroids

def getFurthestCentroids(n,Samples):
    numberOfSamples = len(Samples)
    numOfTuples = len(Samples[0])
    centroids = np.ndarray(shape=(n,numOfTuples),dtype=float)
    
    selectedCentroids = []
    centroidIndex = rd.randint(0,numberOfSamples-1)
    centroids[0] = Samples[centroidIndex]
    selectedCentroids.append(centroidIndex)
        
    for j in range(1,n):
        totalDistance = np.zeros(shape=(numberOfSamples,),dtype=float)
        for k in range(j):
            distance = la.norm(Samples - centroids[k],axis=1)
            totalDistance = totalDistance + distance
        avgDistance = np.true_divide(totalDistance,j)
        avgDistance[selectedCentroids] = -1.00
        centroidIndex = np.argmax(avgDistance)
        centroids[j] = Samples[centroidIndex]
        selectedCentroids.append(centroidIndex)
    return centroids

def kMeans(k,centroids,Samples):
    numberOfSamples = len(Samples)
    numOfTuples = len(Samples[0])
    objFunction = 0
    prev = np.ndarray(shape=(k,numOfTuples))
    clusters = dict()
    
    while not np.array_equal(centroids,prev):
        clusters = dict()
        euclideanNorms = np.asarray([la.norm(Samples- j,axis=1) for j in centroids]).T
        indices = np.argmin(euclideanNorms,axis=1)
        prev = centroids.copy()
        for m in range(len(indices)):
            if not indices[m] in clusters:
                clusters[indices[m]] = []
            clusters[indices[m]].append(Samples[m])
        centroids = np.asarray([np.mean(np.asarray(clusters[p]),axis=0) for p in sorted(list(clusters.keys()))])
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Plot for K='+str(k))
    colors = ['#ff00ff','#6666ff','#00ccff','#00ff99','#009900','#cccc00','#ff9933','#669999','#b35900','#0000ff']
    objectiveFunctionValue = 0          
    for t in sorted(list(clusters.keys())):
        clusterPoints = np.asarray(clusters[t])
        transposeMatrix = clusterPoints.T
        scatterPlot(ax,colors,centroids,transposeMatrix,t)
        distance = clusterPoints - centroids[t]
        euclideanDistance = la.norm(distance, axis=1)
        squaredDistance = np.square(euclideanDistance)
        objectiveFunctionValue += np.sum(squaredDistance)
    objFunction = objectiveFunctionValue
    return objFunction


def plotObjectiveFunction(strategyName,objFunctionValues,MIN_K,MAX_K):
    objPlotFig = plt.figure()
    plotFigure = objPlotFig.add_subplot(1,1,1)
    plotFigure.plot([h for h in range(MIN_K,MAX_K+1)],objFunctionValues)
    plotFigure.set_xlabel('K')
    plotFigure.set_ylabel('Objective Function')
    plotFigure.set_title('Objective Function vs K for ('+str(strategyName)+')')

def scatterPlot(ax,colors,centroids,plotMatrix,index):
    ax.scatter(plotMatrix[0],plotMatrix[1],c=colors[index],s=20)
    ax.scatter(centroids[index][0],centroids[index][1],c='#000000',marker="x",s=50)

def strategy1(MIN_K,MAX_K,Samples):
    objFunctionValues = []
    for i in range(MIN_K,MAX_K+1):
        centroids = getRandomCentroids(i,Samples)
        objFunctionValue = kMeans(i,centroids,Samples)
        objFunctionValues.append(objFunctionValue)
    return objFunctionValues
    
def strategy2(MIN_K,MAX_K,Samples):
    objFunctionValues = []
    for i in range(MIN_K,MAX_K+1):
        centroids = getFurthestCentroids(i,Samples)
        objFunctionValue = kMeans(i,centroids,Samples)
        objFunctionValues.append(objFunctionValue)
    return objFunctionValues

NumpyFile = io.loadmat('AllSamples.mat')
Samples = NumpyFile['AllSamples']
MIN_K = 2
MAX_K = 10

#Plotting the Points Initially
transposeSamples = np.transpose(Samples)
plt.scatter(transposeSamples[0],transposeSamples[1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#Obtaining Objective Function Values for Strategy 1 for two times
strategy1_attempt1_objValues = strategy1(MIN_K,MAX_K,Samples)
strategy1_attempt2_objValues = strategy1(MIN_K,MAX_K,Samples)

#Obtaining Objective Function Values for Strategy 2 for two times
strategy2_attempt1_objValues = strategy2(MIN_K,MAX_K,Samples)
strategy2_attempt2_objValues = strategy2(MIN_K,MAX_K,Samples)

#Plotting Strategy 1 Objective Function Values against respective K
plotObjectiveFunction("Strategy 1",strategy1_attempt1_objValues,MIN_K,MAX_K)
plotObjectiveFunction("Strategy 1",strategy1_attempt2_objValues,MIN_K,MAX_K)

#Plotting Strategy 2 Objective Function Values against respective K
plotObjectiveFunction("Strategy 2",strategy2_attempt1_objValues,MIN_K,MAX_K)
plotObjectiveFunction("Strategy 2",strategy2_attempt2_objValues,MIN_K,MAX_K)