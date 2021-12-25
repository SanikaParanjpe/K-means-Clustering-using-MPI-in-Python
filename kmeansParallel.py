from itertools import cycle,islice
import sys
from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import random as rd
import math
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
import time
from sklearn.cluster import KMeans


"""This function finds the eulicdean distacen between data point and all centroids."""
def FindEuclideanDistance(items, centroids):
    sum = 0
    for i in range(len(centroids)):
        sum += math.sqrt(abs((centroids[i]- items[i])** 2))
    return sum

"""This function is used for data cleaning and preprocessing"""
def DataCleaning(data):
    del data["ocean_proximity"] #not using this feature for clustering
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #impute values where value is missing
    imputer.fit(data[["total_bedrooms"]])
    data["total_bedrooms"] = imputer.transform(data[["total_bedrooms"]])
    # print("Returning from data cleaning")
    return data

"""This function is used to recalulate new local centroids from the datapoints attached to each cluster"""
def recalculate_centroids(clusters, k,Centroids):
    """ Recalculates the centroid position based on the plot """
    centroids =[]
    for i in range(k):
        if clusters[i]:
            centroids.append(np.average(clusters[i], axis=0))
        else:
            centroids.append(Centroids[i])
    return centroids

def CompareCentroids(new_Centroids,Centroids,K):
    meanSqerror = 0
    for i in range(K):
        for j in range(len(Centroids[i])):
            temp = math.sqrt(abs(new_Centroids[i][j] **2 - Centroids[i][j] **2))
            meanSqerror = meanSqerror +temp
    meanSqerror = meanSqerror/K
    # print(meanSqerror)
    return meanSqerror

"""
I have reffered the two functions below from a notebook from Kaggle.
Link: https://www.kaggle.com/benherbertson/california-housing-k-means-clustering
These functions are used for plotting the parallel plot which enables to visualize the centriods after convergence.
"""
def pd_centers(featuresUsed, centers):
    colNames = list(featuresUsed)
    colNames.append('predictedclusters')
    Z = [np.append(A, index) for index, A in enumerate(centers)]
    P = pd.DataFrame(Z, columns=colNames)
    P['predictedclusters'] = P['predictedclusters'].astype(int)
    return P

def parallel_plot(data):
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
    plt.figure(figsize=(15, 8)).gca().axes.set_ylim([-3, +3])
    parallel_coordinates(data, 'predictedclusters', color=my_colors, marker='o')
    plt.savefig('kmeans_centroids.png')

"""End of reffered code"""

if __name__ == '__main__':
    # accept the dataset size from arguments (for weak scaling)
    dataset_size = int(sys.argv[1])

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        #Reading and Preprocessing of data
        dataset = pd.read_csv('housing.csv')
        # print("Dataset -> ", dataset.describe().transpose())
        data = np.array(DataCleaning(dataset))
        dataset = data[:dataset_size][:]

        """Showing Calculation of number of clusters.
        Calculating the number of optimal clusters for the data using WCSS ( Within-Cluster Sum of Square )
        Using the inbuilt k-means function for same."""
        wcss = []
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for num_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
            kmeans.fit(dataset)
            wcss.append(kmeans.inertia_)
        # plot the SSDs for each n_clusters
        plt.plot(wcss)
        plt.savefig('elbow_analysis.png')
        """Here Elbow Analysis is completed and the graph shows that 4 clusters are optimal. Hence using K=4."""

        K = 4
        m = dataset.shape[0]  # number of training examples
        n = dataset.shape[1]  # number of features in the dataset
        Centroids = []
        rd.seed(150)
        for i in range(K):
            rand = rd.randint(0, m - 1)
            # print(dataset[rand])
            Centroids.append(dataset[rand])
        # print("Initial Centroids are: ", Centroids)
        Curr_Centroids = Centroids
    else:
        #initialize K,n,m, error for other ranks
        K = None
        n = None
        m = None
        mean_sq_error = None

    start_time = time.perf_counter() #start timer for time analysis

    K = comm.bcast(K, root=0)
    n = comm.bcast(n, root=0)
    m = comm.bcast(m, root=0)
    # print("K recieved in rank", rank, K)
    # print("n recieved in rank", rank, n)
    # print("m recieved in rank", rank, m)
    comm.Barrier()

    #scatter the datapoints to all processes
    send_data = None
    if rank == 0:
        """
        Below logic for using Scatterv is reffered from below links.
        https://stackoverflow.com/questions/36025188/along-what-axis-does-mpi4py-scatterv-function-split-a-numpy-array
        https://stackoverflow.com/questions/65082585/mpi4py-scatter-a-matrix
        """
        # splitting the datapoints into the number of processes
        split = np.array_split(dataset, size, axis=0)

        split_sizes = []
        for i in range(0, len(split)):
            split_sizes = np.append(split_sizes, len(split[i]))
        #print("split_sizes", split_sizes)

        #store the data in contigous memory location using ravel
        raveled = [np.ravel(arr) for arr in split]

        Columns = split_sizes*n
        displacements = np.insert(np.cumsum(Columns), 0, 0)[0:-1] #calculation of displacement of memory locations
        #print("displacements", displacements)

        serialized_data = np.concatenate(raveled)

    else:
        Columns = None
        displacements = None
        serialized_data = None
        split = None

    Columns = comm.bcast(Columns, root=0)
    displacements = comm.bcast(displacements, root=0)
    split = comm.bcast(split,root=0)

    DataChunk = np.empty(np.shape(split[rank]))

    comm.Scatterv([serialized_data, Columns, displacements, MPI.DOUBLE], DataChunk, root=0)
    """ End of reffered code for Scatterv"""

    comm.Barrier()
    # print('Rank: ', rank, ', datachunk received: ', DataChunk, DataChunk.shape)
    # data chunk sent to all processes

    #broadcast initial centroids to all processes
    if rank != 0:
        Curr_Centroids = np.empty((K, n))
    Curr_Centroids = comm.bcast(Curr_Centroids, root=0)  # broadcast the centroids from rank 0 to all others
    # print('Rank: ', rank, ', centroids received: ', Curr_Centroids)

    start_time_convergence = time.perf_counter()
    while True:
        clusters = {}
        for i in range(K):
            clusters[i] = []
        for item in DataChunk:
            EuclidianDistance = []
            for k in range(K):
                EuclidianDistance.append(FindEuclideanDistance(item, Curr_Centroids[k]))
            clusters[EuclidianDistance.index(min(EuclidianDistance))].append(item)
            # print("EuclidianDistance",EuclidianDistance)
        new_local_Centroids = recalculate_centroids(clusters, K,Curr_Centroids)
        # print('Rank: ', rank, ', new local centroids ', new_local_Centroids)

        if rank != 0:
            comm.send(new_local_Centroids,0,rank)

        if rank == 0:
            all_recv_cent = [new_local_Centroids]
            for i in range(1,size):
                recv_centroids = comm.recv(source=i,tag=i)
                all_recv_cent.append(recv_centroids)
            # print("all_recv_cent",all_recv_cent)

        comm.Barrier()  #using barrier for synchronization

        if rank == 0:
            Next_Centroids = []
            for j in range(len(all_recv_cent[0])):
                temp = []
                for i in range(len(all_recv_cent)):
                    temp.append(all_recv_cent[i][j])
                Next_Centroids.append(np.average(temp, axis=0))
            # print("Next_Centroids", Next_Centroids)
            mean_sq_error = CompareCentroids(Next_Centroids, Curr_Centroids, K)
            Curr_Centroids = Next_Centroids

        Curr_Centroids = comm.bcast(Curr_Centroids, root=0)  # broadcast the centroids from rank 0 to all others
        # print('Rank: ', rank, ',again centroids received: ', Curr_Centroids)

        mean_sq_error = comm.bcast(mean_sq_error, root=0)  # broadcast the error from rank 0 to all others
        # print('Rank: ', rank, ',mean_sq_error received: ', mean_sq_error)

        if mean_sq_error < 0.01:
            break #breaks from all processes if error is less than 0.01

    end_time_convergence = time.perf_counter()
    time_convergence = end_time_convergence - start_time_convergence
    comm.Barrier() #using barrier for synchronization

    total_time_convergence = np.zeros(1)
    total_time_convergence = comm.reduce(time_convergence, op=MPI.SUM, root=0)
    if rank == 0:
        print("Time required for K-means to converge for number of processes : ", size, "is ", total_time_convergence/size)
    print("Done")

    end_time = time.perf_counter()
    # print("TIME", end_time - start_time)
    time_req = end_time - start_time
    comm.Barrier()

    total_time = np.zeros(1)
    total_time = comm.reduce(time_req, op=MPI.SUM, root=0)


    if rank == 0:
        # print("Final Centroids: ", Curr_Centroids)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(Curr_Centroids)   #scaling centroid values for visualization
        P = pd_centers(range(n), scaled) # parallel plot to visualize centroids
        parallel_plot(P)
        print("TIME for number of processes",size ,"is",total_time/size)

"""
References for MPI programming in Python
https://www.kth.se/blogs/pdc/2019/11/parallel-programming-in-python-mpi4py-part-2/
https://rabernat.github.io/research_computing/parallel-programming-with-mpi-for-python.html
https://stackoverflow.com/questions/36025188/along-what-axis-does-mpi4py-scatterv-function-split-a-numpy-array
https://stackoverflow.com/questions/65082585/mpi4py-scatter-a-matrix

Reference for K-means Clustering Algorithm:
https://pythonprogramming.net/k-means-from-scratch-machine-learning-tutorial/
https://www.machinelearningplus.com/predictive-modeling/k-means-clustering/
https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42
"""