from numpy import uint8
###
#
# BIGANN large config - run this on a cloud machine!  Like this:
# `python3 centroids.py config_bigann_small`
# `python3 shard.py config_bigann_small`
#
###

#Random seed for reproducibility of the kmeans clustering (set to None for non-determinism)
RANDOM_SEED=505

#Size of the sample of points examined for during clustering
SAMPLE_SIZE=10000000

#Number of samples per batch
BATCH_SIZE=100000

#Number of centroids to find
S=10000

#Maximum iterations of the kmeans clustering centroid fitter
MAX_ITER=100

#Maximum data points to index (set to None to index everything in the dataset file)
MAX_POINTS=1000000000

#Path to the datafiles
DATA_TYPE=uint8
DATA_FILE="../data/bigann/learn.100M.u8bin"
QUERY_FILE="../data/bigann/query.public.10K.u8bin"

#Path to the index shard data
INDEX_PATH="/datadrive/bigann/shards/"