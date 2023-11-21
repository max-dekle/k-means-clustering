import numpy as np
import random
import math

class Kmeans:
    def __init__(self, X, K, max_iters):
        # Data
        self.X = X
        # Number of clusters
        self.K = K
        # Number of maximum iterations
        self.max_iters = max_iters
        # Initialize centroids
        self.centroids = self.init_centroids()

    def init_centroids(self):
        """
        Selects k random rows from inputs and returns them as the chosen centroids.
        :return: a Numpy array of k cluster centroids, one per row
        """
        unique_rows = np.unique(self.X, axis=0)
        # choose K indices from unique rows
        cent_idx = np.random.choice(unique_rows.shape[0], self.K, replace=False)
        # choose K random rows as centroids
        centroids = unique_rows[cent_idx]
        return centroids

    def euclidean_dist(self, x, y):
        """
        Computes the Euclidean distance between two points, x and y

        :param x: the first data point, a Python numpy array
        :param y: the second data point, a Python numpy array
        :return: the Euclidean distance between x and y
        """
        return np.sqrt(np.sum(np.power(x - y, 2)))

    def closest_centroids(self):
        """
        Computes the closest centroid for each data point in X, returning
        an array of centroid indices

        :return: an array of centroid indices
        """
        data_length = self.X.shape[0]
        centroid_idx = np.zeros(data_length, dtype=int)
        for i, x in enumerate(self.X):
            minimum_dist = float('inf')
            for j, c in enumerate(self.centroids):
                 # calculate distance between datapoint and centroid using euclidean_dist function
                dist = self.euclidean_dist(x, c)
                if dist < minimum_dist:
                    minimum_dist = dist
                    # set the index of the centroid closest to the datapoint as the ith element in centroid_idx
                    centroid_idx[i] = j
        return centroid_idx

    def compute_centroids(self, centroid_indices):
        """
        Computes the centroids for each cluster, or the average of all data points
        in the cluster. Update self.centroids.

        Check for convergence (new centroid coordinates match those of existing
        centroids) and return a Boolean whether k-means has converged

        :param centroid_indices: a Numpy array of centroid indices, one for each datapoint in X
        :return boolean: whether k-means has converged
        """
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.K):
            cluster_points = self.X[centroid_indices == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
        if np.allclose(self.centroids, new_centroids):
            return True
        else:
            self.centroids = new_centroids
            return False

    def run(self):
        """
        Run the k-means algorithm on dataset X with K clusters for max_iters.
        :return: a tuple of (cluster centroids, indices for each data point)
        """
        for i in range(self.max_iters):
            centroid_idx = self.closest_centroids()
            if self.compute_centroids(centroid_idx) == True:
                break
        return self.centroids, centroid_idx

    def inertia(self, centroids, centroid_indices):
        """
        :param centroids - the coordinates that represent the center of the clusters
        :param centroid_indices - the index of the centroid that corresponding data point it closest to
        :return inertia as a float
        """
        inertia = 0.0
        for k in range(self.K):
            idx = np.where(centroid_indices == k)[0]
            c = centroids[k]
            # Compute the squared Euclidean distance between each data point and the centroid
            sq_dist = np.square(self.X[idx] - c)
            sum_sq_dist = np.sum(sq_dist, axis=1)
            inertia += np.sum(sum_sq_dist)
        return inertia
