"""
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
"""
import numpy as np


class KMeans(object):
    def __init__(self, points, k, init="random", max_iters=10000, rel_tol=1e-05):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == "random":
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        N = self.points.shape[0]
        idx = np.random.permutation(N)[:self.K]
        self.centers = self.points[idx].astype(float)
        return self.centers

    def kmpp_init(self):
        """
            Use the intuition that points further away from each other will probably be better initial centers.
            To complete this method, refer to the steps outlined below:.
            1. Sample 1% of the points from dataset, uniformly at random (UAR) and without replacement.
            This sample will be the dataset the remainder of the algorithm uses to minimize initialization overhead.
            2. From the above sample, select only one random point to be the first cluster center.
            3. For each point in the sampled dataset, find the nearest cluster center and record the squared distance to get there.
            4. Examine all the squared distances and take the point with the maximum squared distance as a new cluster center.
            In other words, we will choose the next center based on the maximum of the minimum calculated distance
            instead of sampling randomly like in step 2. You may break ties arbitrarily.
            5. Repeat 3-4 until all k-centers have been assigned. You may use a loop over K to keep track of the data in each cluster.
        Return:
            self.centers : K x D numpy array, the centers.
        Hint:
            You could use functions like np.vstack() here.
        """
        N = self.points.shape[0]
        sample_size = max(1, int(0.01 * N))
        sample_idx = np.random.permutation(N)[:sample_size]
        sample = self.points[sample_idx]
        
        centers = [sample[np.random.randint(sample_size)]] #first center
        
        for _ in range(1, self.K):
            dists = pairwise_dist(sample, np.vstack(centers))
            min_dists = np.min(dists**2, axis=1)
            next_idx = np.argmax(min_dists)
            centers.append(sample[next_idx])
        
        self.centers = np.vstack(centers).astype(float)
        return self.centers

    def update_assignment(self):
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: Do not use loops for the update_assignment function
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison.
        """
        D = pairwise_dist(self.points, self.centers)
        self.assignments = np.argmin(D, axis=1).astype(int)
        return self.assignments

    def update_centers(self):
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        HINT: If there is an empty cluster then it won't have a cluster center, in that case the number of rows in self.centers can be less than K.
        """
        new_centers = []
        for k in range(self.K):
            in_cluster = (self.assignments == k)
            if np.any(in_cluster):
                new_centers.append(self.points[in_cluster].mean(axis=0))
        
        if new_centers:
            self.centers = np.vstack(new_centers).astype(float)
        return self.centers

    def get_loss(self):
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        if self.assignments is None:
            self.assignments = self.update_assignment()
        
        assignments = self.assignments.astype(int)
        assigned_centers = self.centers[assignments]
        diffs = self.points - assigned_centers
        self.loss = np.sum(diffs**2)
        return self.loss

    def train(self):
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster,
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned,
                     pick a random point in the dataset to be the new center and
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference
                     in loss compared to the previous iteration is less than the given
                     relative tolerance threshold (self.rel_tol).
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!

        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.

        HINT: Do not loop over all the points in every iteration. This may result in time out errors
        HINT: Make sure to care of empty clusters. If there is an empty cluster the number of rows in self.centers can be less than K.
        """
        prev_loss = None
        
        for iteration in range(self.max_iters):
            self.assignments = self.update_assignment()
            self.centers = self.update_centers()
            
            # empty cluster handling
            if self.centers.shape[0] < self.K:
                N = self.points.shape[0]
                missing = self.K - self.centers.shape[0]
                extra_idx = np.random.permutation(N)[:missing]
                extras = self.points[extra_idx]
                self.centers = np.vstack([self.centers, extras])
                self.assignments = self.update_assignment()
            
            current_loss = self.get_loss() #check convergence 
            if prev_loss is not None:
                if prev_loss == 0:
                    rel_change = 0 if current_loss == 0 else float('inf')
                else:
                    rel_change = abs(current_loss - prev_loss) / prev_loss
                
                if rel_change < self.rel_tol:
                    break
            prev_loss = current_loss
        
        self.loss = current_loss
        return self.centers, self.assignments, self.loss


def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]

    HINT: Do not use loops for the pairwise_distance function
    """
    x_sq = np.sum(x**2, axis=1)
    y_sq = np.sum(y**2, axis=1)
    sq_dists = x_sq[:, None] + y_sq[None, :] - 2 * x.dot(y.T)
    return np.sqrt(np.maximum(sq_dists, 0))


def fowlkes_mallow(xGroundTruth, xPredicted):
    """
    Args:
        xPredicted : list of length N where N = no. of test samples
        xGroundTruth: list of length N where N = no. of test samples
    Return:
        fowlkes-mallow value: final coefficient value of type np.float64

    HINT: You can use loops for this function.
    HINT: The idea is to make the comparison of Predicted and Ground truth in pairs.
        1. Choose a pair of points from the Prediction.
        2. Compare the prediction pair pattern with the ground truth pair.
        3. Based on the analysis, we can figure out whether it's a TP/FP/FN/FP.
        4. Then calculate fowlkes-mallow value
    """
    n = len(xGroundTruth)
    tp = fp = fn = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            same_gt = (xGroundTruth[i] == xGroundTruth[j])
            same_pred = (xPredicted[i] == xPredicted[j])
            
            if same_gt and same_pred:
                tp += 1
            elif same_gt:
                fn += 1
            elif same_pred:
                fp += 1
    
    if tp == 0:
        return np.float64(0.0)
    
    return np.float64(tp / np.sqrt((tp + fn) * (tp + fp)))
