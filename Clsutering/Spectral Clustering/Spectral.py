
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from scipy import linalg as la
from kmeans import test_kmeans
from RBF import RBF


def RbfKernel(X1, X2):
    sigma = 1.5
    d =np.matrix(abs(np.subtract(X1, X2)))
    squared_d = (np.square(d).sum(axis=1))
    return np.exp(-(squared_d)/(2 * sigma**2))

def similarMatrix(X):
    r = X.shape[0]
    result = np.full((r,r), 0, dtype=np.float)
    for i in range(0,r):
        for j in range(0, r):
            result[i,j] = RbfKernel(X[i, :], X[j, :])
    return result


def degreeMatrix(sim_mat):
    diag = np.array(sim_mat.sum(axis = 1)).flatten()
    return np.diag(diag)

def spectralMatrix(laplacian_mat):
    eigen_vals, eigen_vectors = la.eig(np.matrix(laplacian_mat))
    ind = eigen_vals.real.argsort()[:k]

    result = np.ndarray(shape=(laplacian_mat.shape[0],0))
    for i in range(1, k):
        res_eigen_vec = np.transpose(np.matrix(eigen_vectors[:, ind[i]]))
        result = np.concatenate((result, res_eigen_vec), axis=1)
    return result

def plotClusters(cluster_members, centroid):
    n = len(cluster_members)
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    plt.title("Clustered data")
    for i in range(n):
        col = next(color)
        memberCluster = np.asmatrix(cluster_members[i])
        plt.scatter(np.array(memberCluster[:, 0]).flatten(), np.array(memberCluster[:, 1]).flatten(), marker=".", s=100, c=col)
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    for i in range(n):
        col = next(color)
        plt.scatter(centroid[i, 0], centroid[i, 1], marker="*", s=400, c=col, edgecolors="black")
    plt.show()





def kMeans(dataActu, spectral_data, centroid_Init,k):
    
    cluster_kmeans = test_kmeans(k)
    while(True):
        #assign cluster whose centroid is closest
        clusterMatrix = cluster_kmeans.ncassignData(centroid_Init, spectral_data)
        #assign data to the cluster
        Cluster_data_spect, Cluster_data_Actu = cluster_kmeans.cmassignData(spectral_data, clusterMatrix, dataActu)
        #calculate new centroid
        centroid_new, new_Cent_Act = cluster_kmeans.findCenter(Cluster_data_spect, Cluster_data_Actu, centroid_Init, dataActu)
        
        plotClusters(Cluster_data_Actu,new_Cent_Act)
        #check for convergence
        if((centroid_Init == centroid_new).all()):
            break

        centroid_Init = centroid_new
    return Cluster_data_Actu, new_Cent_Act




if __name__ == "__main__":
    X = np.loadtxt("test1_data.txt", delimiter=" ")
    X = np.loadtxt("test2_data.txt", delimiter=" ")
    X = np.loadtxt("test5.txt", delimiter=" ")
    
    k = int(input("Enter the number of cluster: "))

    
    similarity_mat = similarMatrix(X)
    degree_mat = degreeMatrix(similarity_mat)
    laplacian_mat = degree_mat-similarity_mat
    spectral_mat = spectralMatrix(laplacian_mat)
    
    
    init_centroid = spectral_mat[np.random.choice(spectral_mat.shape[0], k, replace=False)]
    cluster_data, centroid = kMeans(X, spectral_mat, init_centroid ,k)
    plotClusters(cluster_data, centroid)
    print("Data Clustered")