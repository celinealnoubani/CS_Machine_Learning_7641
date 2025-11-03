import numpy as np
from kmeans import pairwise_dist


class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset

    def fit(self):
        """
        Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        See in what order the clusters are being expanded and new points are being checked, recommended to check neighbors of a point then decide whether to expand the cluster or not.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        n_points = len(self.dataset)
        cluster_idx = np.full(n_points, -1)  # noise initialization
        visitedIndices = set()
        C = 0  
        
        for P in range(n_points):
            if P not in visitedIndices:
                visitedIndices.add(P) # label as visited
                
                NeighborPts = self.regionQuery(P)
                if len(NeighborPts) < self.minPts:
                    pass
                else:
                    self.expandCluster(P, NeighborPts, C, cluster_idx, visitedIndices)
                    C += 1
        
        return cluster_idx


    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """
        Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints:
            1. np.concatenate(), and np.sort() may be helpful here. A while loop may be better than a for loop.
            2. Use, np.unique(), np.take() to ensure that you don't re-explore the same Indices. This way we avoid redundancy.
        """
        cluster_idx[index] = C
        NeighborPts = list(neighborIndices)
        i = 0
        
        while i < len(NeighborPts):
            P_prime = NeighborPts[i]
            if P_prime not in visitedIndices:
                visitedIndices.add(P_prime) # label as visited
                
                NeighborPts_prime = self.regionQuery(P_prime)
                if len(NeighborPts_prime) >= self.minPts:
                    for new_neighbor in NeighborPts_prime:
                        if new_neighbor not in NeighborPts:
                            NeighborPts.append(new_neighbor)
            
            if cluster_idx[P_prime] == -1:
                cluster_idx[P_prime] = C
            i += 1

    def regionQuery(self, pointIndex):
        """
        Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        query_point = self.dataset[pointIndex:pointIndex+1]  
        
        distances = pairwise_dist(query_point, self.dataset).flatten()
        indices = np.argwhere(distances <= self.eps).flatten()
        
        return indices

