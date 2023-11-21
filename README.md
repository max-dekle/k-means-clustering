# K-Means Clustering

## K-Means Algorithm
	- Initialization: K random centroids are selected from the input dataset.
	- Point assignment: For each datapoint, the centroid closest to it is selected using the Euclidean distance function.
	- Updating: The centroids are updated by computing the mean of all datapoints assigned to each centroid.
	- Convergence: The above two steps are repeated until the centroids stabilize or the maximum iteration count is reached. 
	- Inertia: The model calculates the sum of square distances of all datapoints and their assigned centroids, otherwise known as inertia.

	The model continuously attempts to minimize the distance between datapoints and their respective centroids, 
	resulting in well-defined clusters once convergence is reached. A lower inertia value represents better
	clustering for each group of datapoints. 

## Clusters
	Each datapoint represents a single pixel in the input image, and the three features used are its respective RGB values. The K-means algorithm clusters similar pixels together based on their color similarity. Each cluster represents a set of pixels that are close in color, and is used to compress the image by replacing all the pixels in the cluster with the same one (in this case, the color defined in the centroid of the cluster). By reducing the number of unique colors in the image, we can also reduce the image dimensions without compromising the overall integrity and quality of the original image. 

	In the song dataset, increasing the number of clusters will result in smaller and more specific song groupings, with each song in the cluster being highly similar to one another. However, having too many clusters might result in over-constraining, and each group might only contain a couple of songs. On the other hand, having a low number of clusters means the groups will contain more songs, but these songs might not be as closely related to each other. 
	
	For the image compression dataset, increasing the amount of clusters we have would mean a higher level of detail in the compressed image, but also a larger file size. But if we have too many clusters, then the compressed image might appear distorted. Having less clusters would mean a smaller image size and simpler image overall, but with less detail and a lower overall quality. 

## Results
	In the code/output directory, we have grouped the songs into a number of clusters based on their acousticness, liveness, and speechiness. Four was found to be the optimal number of clusters for our dataset, as in the elbow plot we can see that the point of inflection occurs at around 4 clusters, at which point adding more clusters has diminishing returns on reducing inertia. 
