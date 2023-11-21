import os
from skimage import io, img_as_ubyte
import numpy as np
from kmeans import Kmeans
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Machine Learning img_compression (Problem 1)')
    parser.add_argument('-d', help='path to data file', default='../data/ml/tiger.jpg')
    parser.add_argument('-o', help='path to output directory', default='output')
    return parser.parse_args()


def image_distance(original, compressed):
    """
    Calculates the pixel element-wise (pixel-by-pixel) difference and normalizes
    the result

    :param original: original image
    :param compressed: compressed image
    :return: image distance value as float
    """
    return np.linalg.norm(abs(original-compressed))


def main():
    np.random.seed(1)
    random.seed(1)
    args = parse_args()
    # Use io to read in the image pixel data
    image = io.imread(args.d)
    # # Uncomment these lines if you'd like to view the original image
    # io.imshow(image)
    # io.show()
    print(f"Read in {args.d}")
    # Get the number of rows and columns
    rows = image.shape[0]
    cols = image.shape[1]
    # Normalize all the pixels by dividing all datapoints by 255
    image = image/255
    # Reshape the image so that the data points have 3 attributes, corresponding to RGB values (for each pixel)
    X = image.reshape(image.shape[0]*image.shape[1], 3)

    # Number of clusters
    K = 16
    # Maximum number of times KMeans should run
    max_iters = 50

    # Create an instance of the KMeans class
    kmeans = Kmeans(X, K, max_iters)
    # Run the instance of the KMeans class on the data X, with K clusters and max_iters
    print("Running k-means")
    centroids, idx = kmeans.run()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    with open(args.o + "/centroids.txt", 'w+') as f:
        for c in centroids:
            f.write("%s\n" % c)

    # Get the compressed version of data X
    X_recovered = centroids[idx]

    print(f"Image Distance: {image_distance(X, X_recovered)}")

    # Reshape X_recovered into a 1-dimensional np array in order for io to display
    X_recovered = np.reshape(X_recovered, (rows, cols, 3))
    io.imshow(X_recovered)
    io.show()

    # Save the compressed version of the original image
    io.imsave(args.o + "/tiger_small.jpg", img_as_ubyte(X_recovered))
    print("Saved compressed image")


if __name__ == '__main__':
    main()
