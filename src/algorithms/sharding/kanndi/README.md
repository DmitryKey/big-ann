KANNDI Sharding Algorithm
=====

The KANNDI (K Approximate Nearest Neighbours DIstance-based) algorithm (read: /ˈkændi/) uses the intuition of multidimensional collocations.
If points in the input dataset are related with respect to a certain distance (Euclidean, cosine, semantic), 
intuitively some points will be closer to each other, than the others. 
This forms a clustered space, where the number of clusters or their density is unknown for any new dataset,
and is highly dependent on the dataset nature: text based embeddings can be distributed differently than, say, image 
descriptors.

![alt text](kanndi.jpeg)

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

Experiment: Indexing
===

Dataset: BIGANN, uint8, 128 dimensions, L2 distance
Size: 100M (sample)

The distance `d` was approximated with the median of all pair-wise distances in a sample. 
For BIGANN sample we get d=534.8055721474861 over first 10000 points.

```
Got the input data matrix: rows = 10000, cols = 128
Points: [[ 0  0  0 ... 14 10  6]
 [65 35  8 ...  0  0  0]
 [ 0  0  0 ...  1  0  0]
 ...
 [ 7  5  7 ... 45 16 23]
 [38  0  0 ... 27  4  8]
 [ 0  0  2 ... 30  3  0]]
Distances: {} [521.67231094 245.12853771 502.93240102 ... 547.31526564 536.60600071
 500.64957805]
Median distance: {} 534.8055721474861
```

The algorithm maintains the processed points (either part of the current shard or already saturated shard).
Therefore, the algorithm converges quite rapidly (complexity analysis is due).

Some of the first iterations show:

```
Size of the current shard after going through the current batch: 0
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000} with 13 keys
Expected shard size: 1000000.0

Processing index=30000000
going inside inner loop by j over current batch of points, skipping the seed point
Seed point for shard 30000000: [  2  57 112  12   2   0   0   0   7  32  61  22  28  13  20  23  20  28
 108  23   4   6  49  84   4  43  94  17   0   2  13  10  14  89 112  18
   2   5   4   9  19  77 112  36   3  13   6  14 112  78 112  22   1   4
   6  17  23  21  42  36  20  16  16  11  22  10  17   7   8  95 112  72
  42  10  10  22  46  53  33  85 112  37  10   8   7   9   5  54  22  15
  14  44  11   5  23  11   0   0   0   4  34  95 112   6   8   3   8  27
  93 101  19  10  49  82  30  14   7   2   3  13  19  28  31  44   6   3
  12   9]
Size of the current shard after going through the current batch: 459779
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000} with 13 keys
Expected shard size: 1000000.0

Processing index=31000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 921375
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000} with 13 keys
Expected shard size: 1000000.0

Processing index=32000000
going inside inner loop by j over current batch of points, skipping the seed point
Saturated shard with id=13. Building HNSW index for it..
Done
```

While in later stages the algorithm tends to process more batches to saturate the shard, each 
batch (of 1M points in this experiment) gets processed more rapidly (TODO: add time measurements):

```
Seed point for shard 67000000: [ 23  69  57   0   0   0   0   3   4  45  69   2   0   1   8   6   0   1
  56  50  12   3   3   2   0   0   7 118  44   8  14   0  16  81 120   0
   0   0   0   0 150 101  67   2   0   2  28  38  18   1  37  27   4  23
 150  29   0   0   4  32  10  15 150  22  42  11   4   0   0   0   1   1
 150  39   0   0   0   0   7  24 112   4   0   0   0   0  79  29   0   0
   0   3   4   1  73   7  18   0   0   0   0  31  81  29 150   4   0   0
   0   0  19 136  88   2   0   2   0   0   0  17   0   0   0  16   7   0
   0   0]
Size of the current shard after going through the current batch: 110450
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=68000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 133458
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=69000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 153505
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=70000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 296976
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=71000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 336812
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=72000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 455706
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=73000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 558008
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=74000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 673878
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=75000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 870342
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=76000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 888067
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=77000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 909704
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000} with 77 keys
Expected shard size: 1000000.0

Processing index=78000000
going inside inner loop by j over current batch of points, skipping the seed point
Saturated shard with id=77. Building HNSW index for it..
```

If algorithm gets stuck on a starving shard, its points will get pushed to a special_shard and new seed points it chosen:

```
Processing index=99000000
going inside inner loop by j over current batch of points, skipping the seed point
Size of the current shard after going through the current batch: 1
Shards built so far: {0: 1000000, 1: 1000000, 2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000, 6: 1000000, 7: 1000000, 8: 1000000, 9: 1000000, 10: 1000000, 11: 1000000, 12: 1000000, 13: 1000000, 14: 1000000, 15: 1000000, 16: 1000000, 17: 1000000, 18: 1000000, 19: 1000000, 20: 1000000, 21: 1000000, 22: 1000000, 23: 1000000, 24: 1000000, 25: 1000000, 26: 1000000, 27: 1000000, 28: 1000000, 29: 1000000, 30: 1000000, 31: 1000000, 32: 1000000, 33: 1000000, 34: 1000000, 35: 1000000, 36: 1000000, 37: 1000000, 38: 1000000, 39: 1000000, 40: 1000000, 41: 1000000, 42: 1000000, 43: 1000000, 44: 1000000, 45: 1000000, 46: 1000000, 47: 1000000, 48: 1000000, 49: 1000000, 50: 1000000, 51: 1000000, 52: 1000000, 53: 1000000, 54: 1000000, 55: 1000000, 56: 1000000, 57: 1000000, 58: 1000000, 59: 1000000, 60: 1000000, 61: 1000000, 62: 1000000, 63: 1000000, 64: 1000000, 65: 1000000, 66: 1000000, 67: 1000000, 68: 1000000, 69: 1000000, 70: 1000000, 71: 1000000, 72: 1000000, 73: 1000000, 74: 1000000, 75: 1000000, 76: 1000000, 77: 1000000, 78: 1000000, 79: 1000000, 80: 1000000, 81: 1000000, 82: 1000000, 83: 1000000, 84: 1000000, 85: 1000000} with 86 keys
Expected shard size: 1000000.0
!!! After going through the whole dataset, the shard did not saturate, at size: 1
!!! Appended to the special_shard, its running size: 10
Processing index=0
going inside inner loop by j over current batch of points, skipping the seed point
Seed point for shard 0: [ 41  38  21  17  42  71  60  50  11   1   2  11 109 115   8   4  27   8
   5  22  11   9   8  14  20  10   4  33  12   7   4   1  18 115  95  42
  17   1   0   0  19   6  46 115  91  16   0   7  66   7   4  15  12  32
  91 109  12   3   1   8  21 115  96  17   1  51  78  14   0   0   0   0
  50  40  62  53   0   0   0   3 115 115  40  12   6  13  25  65   7  30
  51  65 110  92  25   9   0   1  13   0   0   0   0   0   4  22  11   1
   0   0   0   0  13 115  48   1   0   0   0   0   0  36 102  63  11   0
   0   0]
```

TODO: this part can be improved by choosing the next seed point smarter, say with approximating the probability 
of a shard "centroid". One way to do this is to measure the median distance over a sample of remaining points and
readjust the distance `d`.

