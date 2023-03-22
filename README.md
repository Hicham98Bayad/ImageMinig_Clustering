Clustering consists of grouping (partitioning) objects into homogeneous groups (packets).
A cluster is a collection of objects (elements) that share common or too close characteristics and that are dissimilar to objects in other clusters.
Clustering consists of a grouping (partitioning) of objects into homogeneous groups (packages) (clusters).<br>
A cluster is a collection of objects (elements) that share common or too close characteristics and that are dissimilar to objects in other clusters.



<h1> 3- Implementation </h1>
In ImageMining, we have two ways to apply (use) Clustering.
  

1. Group an image database into subsets. For example, we have an image database containing a multitude of image types; aerial images, images of animals, images of vehicles, ... and we want to group each of the images in the database into its category.
2. Segment images.


In this workshop, we will first cluster two image databases<br>

1. Images to be grouped into 2 classes<br>
2. Image to be grouped into 4 classes<br>

The clustering results will be saved in a *.csv file respecting the following form:

image_name,class
image1.jpg,1
image2.jpg,2
image3.jpg,1

Thereafter we will segment images using a clustering algorithm and also we will apply supervised classification for image segmentation.


# <h1>The clustering algorithms used</h1>
There are a multitude of clustering algorithms that can be used in the case of Image mining:

1- Centroid-based methods such as k-means or k-medoid algorithms <br>
2- Hierarchical grouping methods<br>
3- Expectation-Maximization Algorithm (EM)<br>
5- Fuzzy C-means clustering algorithm (FCM) <br>
    …<br>
