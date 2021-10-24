from numpy import float32
###
#
# BIGANN small test config! Run it on your personal machine like this:
# `python3 centroids.py config_bigann_small`
# `python3 shard.py config_bigann_small`
#
###

#Random seed for reproducibility of the kmeans clustering (set to None for non-determinism)
RANDOM_SEED=505

#Size of the sample of points examined for during clustering
SAMPLE_SIZE=100000

#Number of samples per batch
BATCH_SIZE=5000

#Number of centroids to find
S=100

#Maximum iterations of the kmeans clustering centroid fitter
MAX_ITER=25

#Maximum data points to index (set to None to index everything in the dataset file)
MAX_POINTS=20000

#Path to the datafiles
DATA_TYPE=float32
DATA_FILE="../data/deep1b/base.1B.fbin.crop_nb_10000000"
QUERY_FILE="../data/deep1b/query.public.10K.fbin"

#Path to the centroid index shard data
INDEX_PATH="../data/shards/"