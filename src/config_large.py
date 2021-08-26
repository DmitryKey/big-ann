#Random seed for reproducibility of the kmeans clustering (set to None for non-determinism)
RANDOM_SEED=505

#Size of the sample of points examined for during clustering
SAMPLE_SIZE=5000000

#Number of samples per batch
BATCH_SIZE=100000

#Number of centroids to find
S=10000

#Maximum iterations of the kmeans clustering centroid fitter
MAX_ITER = 25

#Maximum data points to index (set to None to index everything in the dataset file)
MAX_POINTS = 2000000