# -*- coding: utf-8 -*-
import scipy.io as io
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from numpy import linalg as la
import sys

NumpyFile = io.loadmat('AllSamples.mat')
Samples = NumpyFile['AllSamples']
MIN_K = 10
MAX_K = 10

#def cluster(MIN_K,MAX_K,Samples):
objFunctions = []
for i in range(MIN_K,MAX_K+1):
    print("Working on k="+str(i))
    numberOfSamples = len(Samples)
    numOfTuples = len(Samples[0])
    centroids = np.ndarray(shape=(i,numOfTuples),dtype=float)
        
    prev = np.ndarray(shape=(i,numOfTuples))
    clusters = dict()
    selectedCentroids = []
    centroidIndex = rd.randint(0,numberOfSamples-1)
    centroids[0] = Samples[centroidIndex]
    selectedCentroids.append(centroidIndex)
        
    for j in range(1,i):
        totalDistance = np.zeros(shape=(numberOfSamples,),dtype=float)
        for k in range(j):
            print(k)
            distance = la.norm(Samples - centroids[k],axis=1)
            totalDistance = totalDistance + distance
        avgDistance = np.true_divide(totalDistance,j)
        avgDistance[selectedCentroids] = -1.00
        centroidIndex = np.argmax(avgDistance)
        centroids[j] = Samples[centroidIndex]
        selectedCentroids.append(centroidIndex)