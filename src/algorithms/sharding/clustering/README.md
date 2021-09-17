K-Means based sharding algorithm
=====

This sharding algorithm is based on K-Means:

1. Cluster the input dataset using K-Means algorithm into `M` clusters (also shards).
2. For each centroid create a new shard graph.
3. The shard is an HNSW graph with neighborhoods of the parent centroid.
4. The shard is persisted to disk for each addition.
5. The shard is loaded from disk and searched when a query is in its centroid neighborhood.

Experiments
===

These settings took 7 minutes on my macbook pro with other stuff running to fit KMeans:
RANDOM_SEED = 505
SAMPLE_SIZE = 100000
M = 1000
MAX_ITER = 50
BATCH_SIZE = 1000000
"""

"""
The idea is to go *very* wide with the clustering, to increase the number of shards
For 10k centroids there are 10k shards (each with 100k vectors)
For 100k centroids there are 100k shards (each with 10k vectors)
For 1m centroids there are 1m shards (each with 1k vectors)