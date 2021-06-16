import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

class test_kmeans():

   def __init__(self, k):
        self.k = k
        
   def findCenter(self, Cluster_data_spect, Cluster_data_Actu, centroid_Init, dataOri):
	   new_Cent_Spect = np.ndarray(shape=(0, centroid_Init.shape[1]))
	   new_Cent_Act = np.ndarray(shape=(0, dataOri.shape[1]))

	   for i in range(0,self.k):
		   centroidClusterSpect = np.asmatrix(Cluster_data_spect[i]).mean(axis=0)
		   centroidClusterOri = np.asmatrix(Cluster_data_Actu[i]).mean(axis=0)
		   new_Cent_Spect = np.concatenate((new_Cent_Spect, centroidClusterSpect), axis=0)
		   new_Cent_Act = np.concatenate((new_Cent_Act, centroidClusterOri), axis=0)

	   return new_Cent_Spect, new_Cent_Act


   def ncassignData(self, centroid_Init, spectralData):
       euclidean_Matrix = np.ndarray(shape=(spectralData.shape[0], 0))
	   
       for i in range(0, self.k):
          c= np.repeat(centroid_Init[i, :], spectralData.shape[0], axis=0)
          d = abs(np.subtract(spectralData, c))
          temp_euclid = np.sqrt(np.square(d).sum(axis = 1))
          euclidean_Matrix = np.concatenate((euclidean_Matrix, temp_euclid), axis=1)
           
       clusterMatrix = np.array(np.argmin(euclidean_Matrix, axis = 1)).flatten()
       
       return clusterMatrix
   
    

   def cmassignData(self, spectralData, clusterMatrix, dataOri):
       Cluster_data_spect = [[] for i in range(self.k)]
       Cluster_data_Actu = [[] for i in range(self.k)]
       iteration = spectralData.shape[0]
       i = 0
       
       while(i < iteration):
           l1=np.array(spectralData[i, :]).flatten()
           Cluster_data_spect[clusterMatrix[i]].append(l1)
           l2=np.array(dataOri[i, :]).flatten()
           Cluster_data_Actu[clusterMatrix[i]].append(l2)
           i = i + 1
       return Cluster_data_spect, Cluster_data_Actu 
    
		   
			
	   
       

		