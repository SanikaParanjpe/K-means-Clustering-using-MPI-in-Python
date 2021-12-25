# K-means-Clustering-using-MPI-in-Python
The topic that is proposed in this project is K-means Clustering Algorithm which is parallelized using  Message passing interface(MPI) in Python. 

In the sequential approach, most of the time that is taken in the algorithm is during the assignment of 
clusters to the data points and the recalculation of new centroids for the next iteration. This execution 
time can be reduced by parallelizing this process.                      
First step is to divide the number of data points into the number of process. Let D be the size of data points
and n be the number of processes then each process will work on D/n data points.[5] Every process has 
information about the centroids which are broadcasting at every iteration. Each process works in data 
points scattered to it and finds the local centroids. The processes find the Euclidean distance from each 
data point under it to each centroids and takes an average of the data points under each cluster to find 
the local centroids. These local centroids are sent to root and averaged to find new global centroids. This 
algorithm is implemented using mpi4py library available for Message passing interface in Python.
