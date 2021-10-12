import sys
from enum import Enum

from util.utils import read_fbin, read_bin, get_total_nvecs_fbin, get_total_dim_fbin, add_points
from numpy import linalg
from statistics import median
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist


# desired number of shardsCreates a new shard graph for a centroid shard
# The shard is an HNSW graph with neighborhoods of the parent centroid.
# The shard is persisted to disk for each addition.
# The shard is loaded from disk and searched when a query is in its centroid neighborhood.
M = 100

# target maximum distance between points to fall inside a shard
DIST_MULTIPLIER = 1

# number of times to sample the input dataset to approximate the dist_max
# SAMPLE_TIMES = 10
# size of the sample of points examined linearly during max dist computation
SAMPLE_SIZE = 1000

# batch size for reading points from the input file during the sharding algorithm
BATCH_SIZE = 1000000


# expected: 1 280 000 008
# file size: 1 280 0 000 008
def compute_median_dist(data_file: str, sample_size: int = SAMPLE_SIZE)->float:
    #points = read_fbin(data_file, start_idx=0, chunk_size=sample_size)
    points = read_bin(data_file, dtype=np.uint8, start_idx=0, chunk_size=sample_size)
    # points = read_bin(filename=data_file, dtype=np.float32, start_idx=0, chunk_size=sample_size)
    num_rows, num_cols = points.shape
    print("Got the input data matrix: rows = {}, cols = {}".format(num_rows, num_cols))
    print("Points: {}".format(points))

    class DistMethod(Enum):
        METHOD_NUMPY = 1,
        METHOD_PAIRWISE_LOOP = 2,
        SPATIAL_DISTANCE_MATRIX = 3

    dists = []

    method = DistMethod.SPATIAL_DISTANCE_MATRIX

    if method == DistMethod.METHOD_NUMPY:
        # Method 1: does not work: computes inf on the diagonal and zeroes elsewhere in the resulting matrix
        # dists = np.sqrt(np.sum((points[None, :] - points[:, None])**2, -1))
        dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    elif method == DistMethod.METHOD_PAIRWISE_LOOP:
        # Method 2: O(Nˆ2) iteration
        # dists = []
        #for i in range(0, num_rows):
        #    for j in range(0, num_rows):
        #        dist = linalg.norm(points[i] - points[j])
        #        dists.append(dist)

        dists = [linalg.norm(points, 'fro')]
    elif method == DistMethod.SPATIAL_DISTANCE_MATRIX:
        #dists = distance_matrix(points, points)
        dists = pdist(points)

    print("Distances: {}", dists)
    median_dist = median(dists.flatten())
    print("Median distance: {}", median_dist)

    if median_dist == np.inf:
        print("Distance computation failed")
        exit(0)

    return DIST_MULTIPLIER * median_dist


# objective function | loss function like in K-Means
def shard_by_dist(data_file: str, dist: float, output_index_path: str, shards_m: int = M):
    VERBOSE = False
    # set of integer order ids of each point that was already placed into a shard => processed
    processed_point_ids = set()
    complete_shards = 0

    total_num_elements = get_total_nvecs_fbin(data_file)
    # dimensionality = get_total_dim_fbin(data_file)
    print(f"Total number of points to process: {total_num_elements}")
    print(f"Reading data from {data_file} in {BATCH_SIZE} chunks")

    range_upper = total_num_elements
    print(f"range_upper={range_upper}")

    # map from shard id to number of elements
    shards = {}

    # expected number of elements per shard
    expected_shard_size = total_num_elements / shards_m

    print("Expected shard size: {}".format(expected_shard_size))

    # get the seed point
    #points = read_fbin(data_file, start_idx=0, chunk_size=1)
    points = read_bin(data_file, dtype=np.uint8, start_idx=0, chunk_size=1)
    seed_point_id = 0
    seed_point = points[seed_point_id]
    print("Seed point for shard {}: {}".format(seed_point_id, seed_point))
    # remember the seed point
    processed_point_ids.add(seed_point_id)
    shard = [seed_point]
    shard_ids = [seed_point_id]

    need_seed_update = False

    # holds points that do not form a complete shard
    special_shard = []

    # number of batches, during which this shard is not growing
    # TODO

    # repeat, while number of shards did not reach the target level M
    while complete_shards < shards_m:
        complete_shards = len(shards.keys())

        # step through the dataset with batch by batch
        for i in range(0, range_upper, BATCH_SIZE):
            print(f"Processing index={i}", flush=True)
            #points = read_fbin(data_file, start_idx=i, chunk_size=BATCH_SIZE)
            points = read_bin(data_file, dtype=np.uint8, start_idx=i, chunk_size=BATCH_SIZE)

            print("going inside inner loop by j over current batch of points, skipping the seed point", flush=True)
            for j in range(0, points.shape[0]):
                if j == seed_point_id:
                    continue
                # id of the shard candidate is a combination of the running i-th batch and offset j within it
                candidate_point_id = i + j
                if candidate_point_id not in processed_point_ids:
                    # update seed point?
                    if need_seed_update:
                        seed_point = points[j]
                        print("Seed point for shard {}: {}".format(i, seed_point))
                        shard = [seed_point]
                        shard_ids = [i]
                        need_seed_update = False
                    else:
                        # seed is up to date and we continue building the shard
                        dist_j = distance_matrix(np.array([seed_point]), np.array([points[j]]))
                        if VERBOSE:
                            print("got dist between seed_point and points[j]: {}".format(dist_j))
                        if dist_j <= dist:
                            if VERBOSE:
                                print("Got a neighbor!")
                            processed_point_ids.add(candidate_point_id)
                            shard.append(points[j])
                            shard_ids.append(candidate_point_id)

                # check if we saturated the shard
                if len(shard) == expected_shard_size:
                    print("Saturated shard with id={}. Building HNSW index for it..".format(i))
                    add_points(output_index_path, str(i), shard_ids, shard)
                    print("Done")
                    shards[i] = len(shard)
                    need_seed_update = True
                    break

            print("Size of the current shard after going through the current batch: {}".format(len(shard)), flush=True)
            print("Shards built so far: {} with {} keys".format(shards, len(shards.keys())), flush=True)
            print("Expected shard size: {}".format(expected_shard_size), flush=True)

            # check if we saturated the shard
            if len(shard) == expected_shard_size:
                add_points(output_index_path, str(i), shard_ids, shard)
                shards[i] = len(shard)

        # we reached the end of the whole dataset and can stash existing points into some "special shard"
        if len(shard) < expected_shard_size:
            print("After going through the whole dataset, the shard did not saturate, at size: {}".format(len(shard)), flush=True)
            special_shard.append(shard)
            print("Appended to the special_shard, its running size: {}".format(len(special_shard)))
            # request to update seed point
            need_seed_update = True

    print("Processed this many points: {}".format(len(processed_point_ids)), flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Expect two params: (1) an input points file (2) an output index path")
        exit(0)

    points_file = sys.argv[1]
    output_index_path = sys.argv[2]

    computed_dist_max = compute_median_dist(points_file)
    print(f"computed {computed_dist_max}")

    shard_by_dist(points_file, computed_dist_max, output_index_path)
