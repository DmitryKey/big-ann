DKANN Sharding Algorithm
=====

The DKANN Algorithm uses the intuition of multidimensional collocations. If points in the input dataset are related with
respect to a certain distance (Euclidean, cosine, semantic), intuitively some points will be closer to each other, than
the others. This forms a clustered space, where the number of clusters or their density is unknown for any new dataset,
and is highly dependent on the dataset nature: text based embeddings can be distributed differently than, say, image 
descriptors.

The core of this algorithm is the following:
1. Compute an average distance `d` between any pair of points in a sample of the dataset.
2. Scan the dataset and group points that are not further away from each other than `d`.
3. Define a maximum size `v` of each group, such that we will have a desired number of groups (shards).
4. If a group of points grows beyond `v`, then new points are not admitted to it anymore and pushed to form a new group.
5. The algorithm converges, once all points found their group.

A bit more formally and detailing both indexing and searching algorithms for a 1 Billion (1B) data points:

INDEXING
===

1. Set `M` as the number of desired shards.
2. Set `d` = as the target maximum distance.
3. Make several passes over monotonically decreasing in size input data set, always starting from the first point in the input list.
4. If `dist(pi, pj)` < `d` => join `pi` and `pj` in the same shard.
5. If `|SHARDx|` >= `1B/M`, mark this shard as "complete" and exclude from adding new points.
6. In the end of the process we should have exactly `M` shards with at most `1B/M` points in them.
7. Some shards might be "starving", which will reflect absence of clusters. However, we still have an upper bound for the size of each shard, which is important.
8. Use the HNSW algorithm to surface the entry point to each shard.

SEARCHING
===
1. For input multidimensional point `p*` make a pass over all entry points -- exactly `M` of them.
2. Select the closest one (or closest `T` ones).
3. Use HNSW to find the top `k` in each shard (or that single shard -- debatable).
4. Form a list from all the shards that participated in the search and re-sort the list with respect to the true distance from `p*`.
5. Return top 10.

