import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from kmeans import Kmeans
import argparse
import json
import random
import pandas as pd
from scipy import stats

from mpl_toolkits.mplot3d import Axes3D  # Not used but needed to make 3D plots


feature_columns = ["artist", "title", "acousticness", "speechiness", "liveness"]
MAX_CLUSTERS = 10
cmap = cm.get_cmap('tab10', MAX_CLUSTERS)


def parse_args():
    """
    Parses the CLI args with argparse
    :return: args from command line
    """
    parser = argparse.ArgumentParser(description='Machine Learning song_clustering (Problem 2)')
    parser.add_argument('-d', help='path to data file', default='../data/ml/spotify.csv')
    parser.add_argument('-o', help='path to output data directory', default='output')
    return parser.parse_args()


def read_data(path):
    """
    Reads the data at the provided files path. Performs some pre-processing
    - removes outliers (data points that are more than 3 stdev away along any
      feature column
    - removes duplicates

    :param path: path to dataset
    :return: raw data, raw music data (only numeric columns)
    """
    # Load the data set into a 2D numpy array
    with open(path) as data_file:
        data = pd.read_csv(data_file)[feature_columns].to_numpy()

    music_data = data[:, 2:].astype(np.float)
    # Calculate z-score for all data points (how many standard deviations away from mean) for each column
    z = np.abs(stats.zscore(music_data))
    # Find all the rows where all values in each row have a z-score less than 3
    ind = np.all((z < 3), axis=1)

    data = data[ind]
    music_data = music_data[ind]
    return data, music_data


def export_centroid_idx(centroids, idx, centroids_sklearn, idx_sklearn):
    """
    Exports the centroid centers and centroid indices for each data point
    :param centroids: numpy ndarray of cluster centers
    :param idx: centroid indices for each data point
    :param centroids_sklearn: numpy ndarray of cluster centers from sklearn
    :param idx_sklearn: centroid indices for each data point from sklearn
    """
    output = {"centroids": centroids.tolist(),
              "indices": idx.tolist(),
              "centroids_sklearn": centroids_sklearn.tolist(),
              "indices_sklearn": idx_sklearn.tolist()}
    with open(output_dir + "/song_clusters.json", 'w') as outfile:
        json.dump(output, outfile)


def visualize_songs_clusters(data, centroids=None, centroid_indices=None,
                             is_lib_kmean=False):
    """
    Visualizes the song data points and (optionally) the calculated k-means
    cluster centers.
    Points with the same color are considered to be in the same cluster.

    Optionally providing centroid locations and centroid indices will color the
    data points to match their respective cluster and plot the given centroids.
    Otherwise, only the raw data points will be plotted.

    :param data: 2D numpy array of song data
    :param centroids: 2D numpy array of centroid locations
    :param centroid_indices: 1D numpy array of centroid indices for each data point in data
    :return:
    """
    def plot_songs(fig, color_map=None):
        x, y, z = np.hsplit(data, 3)
        fig.scatter(x, y, z, c=color_map)

    def plot_clusters(fig):
        x, y, z = np.hsplit(centroids, 3)
        fig.scatter(x, y, z, c="black", marker="x", alpha=1, s=200)

    plt.clf()
    cluster_plot = centroids is not None and centroid_indices is not None

    ax = plt.figure(num=1).add_subplot(111, projection='3d')
    colors_s = None

    if cluster_plot:
        if max(centroid_indices) + 1 > MAX_CLUSTERS:
            print(f"Error: Too many clusters. Please limit to fewer than {MAX_CLUSTERS}.")
            exit(1)
        colors_s = [cmap(l / MAX_CLUSTERS) for l in centroid_indices]
        plot_clusters(ax)

    plot_songs(ax, colors_s)

    ax.set_xlabel(feature_columns[2])
    ax.set_ylabel(feature_columns[3])
    ax.set_zlabel(feature_columns[4])

    plot_name = "/data"
    plot_name = plot_name + "_clusters" if cluster_plot else plot_name + "_raw"
    plot_name = plot_name + "_sklearn" if is_lib_kmean else plot_name

    ax.set_title(plot_name[1:])
    
    # Helps visualize clusters
    plt.gca().invert_xaxis()
    plt.savefig(output_dir + plot_name + ".png")
    plt.show()


def elbow_point_plot(cluster, errors):
    """
    This function helps create a plot representing the tradeoff between the
    number of clusters and the inertia values.

    :param cluster: 1D np array that represents K (the number of clusters)
    :param errors: 1D np array that represents the inertia values
    """
    plt.clf()
    plt.plot(cluster, errors)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('elbow_plot')
    plt.savefig(output_dir + "/elbow_plot.png")
    plt.show()


def min_max_scale(data):
    """
    Pre-processes the data by performing MinMax scaling.

    MinMax scaling prevents different scales of the data features from
    influencing distance calculations.

    MinMax scaling is performed by
        X_new = (X - X_min) / (X_max - X_min),

    where X_new is the newly scaled value, X_min is the minimum and X_max is the
    maximum along a single feature column.

    :param data: 2D numpy array of raw data
    :return: preprocessed data
    """
    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)

    scaled_data = (data - min_data) / (max_data - min_data)

    return scaled_data


def sk_learn_cluster(X, K):
    """
    Performs k-means clustering using library functions (scikit-learn). You can
    experiment with different initialization settings, but please initialize
    without any optional arguments (other than n_clusters) before submitting.

    :param X: 2D np array containing features of the songs
    :param K: number of clusters
    :return: a tuple of (cluster centroids, indices for each data point)
    """
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(X)
    
    cent = kmeans.cluster_centers_
    idx = kmeans.labels_

    return (cent, idx)


def cluster_songs(music_data, max_iters=300):
    """
    :param music_data: 2D numpy array of music data
    :param max_iters: max number of iterations to run k-means
    :return:
        centroids: calculated cluster centroids from YOUR k-means object
        idx: centroid indices for each data point from YOUR k-means object
        centroids_sklearn: calculated cluster centroids from the sklearn k-means object
        idx_sklearn: centroid indices for each data point from the sklearn k-means object
    """
    centroids = None
    idx = None

    centroids_sklearn = None
    idx_sklearn = None

    scaled_data = min_max_scale(music_data)

    visualize_songs_clusters(scaled_data, centroids, idx)

    errors = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, max_iter=max_iters)
        kmeans.fit(scaled_data)
        errors.append(kmeans.inertia_)
    elbow_point_plot(K, errors)

    kmeans = Kmeans(scaled_data, 4, max_iters)
    centroids, idx = kmeans.run()

    visualize_songs_clusters(scaled_data, centroids, idx, is_lib_kmean=False)

    centroids_sklearn, idx_sklearn = sk_learn_cluster(scaled_data, 4)
    visualize_songs_clusters(scaled_data, centroids_sklearn, idx_sklearn, is_lib_kmean=True)

    return centroids, idx, centroids_sklearn, idx_sklearn


def main():
    """
    Main function for running song clustering. You should not have to change
    anything in this function.
    """
    # Set random seeds
    np.random.seed(0)
    random.seed(0)

    data, music_data = read_data(data_dir)
    max_iters = 300  # Number of times the k-mean should run

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    centroids, idx, centroids_sklearn, idx_sklearn = cluster_songs(music_data, max_iters=max_iters)
    export_centroid_idx(centroids, idx, centroids_sklearn, idx_sklearn)


if __name__ == '__main__':
    # Make args global variable
    args = parse_args()
    
    data_dir = args.d
    output_dir = args.o

    main()
