import gc
from enum import Enum

from intervaltree import IntervalTree, Interval
from util.utils import read_bin, get_total_nvecs_fbin, add_points, Shard, read_fbin, \
    is_number_in_interval_tree, intervals_extract
from numpy import linalg
from statistics import median
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist
# import tracemalloc
import argparse


# desired number of shardsCreates a new shard graph for a centroid shard
# The shard is an HNSW graph with neighborhoods of the parent centroid.
# The shard is persisted to disk for each addition.
# The shard is loaded from disk and searched when a query is in its centroid neighborhood.
M = 1000

# target maximum distance between points to fall inside a shard
DIST_MULTIPLIER = 2

# size of the sample of points examined linearly during max dist computation
SAMPLE_SIZE = 10000

# batch size for reading points from the input file during the sharding algorithm
BATCH_SIZE = 1000000

# with this minimum saturation % we will save the current shard, if it did not grow further
SHARD_SATURATION_PERCENT_MINIMUM = 75

VERBOSE = False


# expected: 1 280 000 008
# file size: 1 280 0 000 008
def compute_median_dist(points)->float:
    num_rows, num_cols = points.shape
    print("Got the input data matrix: rows = {}, cols = {}".format(num_rows, num_cols))
    print("Points: {}".format(points))

    class DistMethod(Enum):
        METHOD_NUMPY = 1,
        METHOD_PAIRWISE_LOOP = 2,
        SPATIAL_DISTANCE_MATRIX = 3,
        PDIST = 4

    dists = []

    method = DistMethod.PDIST

    if method == DistMethod.METHOD_NUMPY:
        # Method 1: does not work: computes inf on the diagonal and zeroes elsewhere in the resulting matrix
        # dists = np.sqrt(np.sum((points[None, :] - points[:, None])**2, -1))
        dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    elif method == DistMethod.METHOD_PAIRWISE_LOOP:
        # Method 2: O(Nˆ2) iteration
        for i in range(0, num_rows):
            for j in range(0, num_rows):
                dist = linalg.norm(points[i] - points[j])
                dists.append(dist)
        dists = [linalg.norm(points, 'fro')]
    elif method == DistMethod.SPATIAL_DISTANCE_MATRIX:
        dists = distance_matrix(points, points)
    elif method == DistMethod.PDIST:
        dists = pdist(points)

    print("Distances: {}", dists, flush=True)
    median_dist = median(dists)
    print("Median distance: {}", median_dist, flush=True)

    if median_dist == np.inf:
        print("Distance computation failed")
        exit(0)

    return median_dist


# objective function | loss function like in K-Means
def shard_by_dist(data_file: str, dist: float, output_index_path: str, dtype:np.dtype, shards_m: int = M):
    #tracemalloc.start()

    # set of integer order ids of each point that was already placed into a shard => processed
    processed_point_ids = set()
    processed_point_id_interval_tree = IntervalTree()

    total_num_elements = get_total_nvecs_fbin(data_file)
    print(f"Total number of points to process: {total_num_elements}", flush=True)
    print(f"Reading data from {data_file} in {BATCH_SIZE} chunks", flush=True)

    range_upper = total_num_elements
    print(f"range_upper={range_upper}")

    # map from globally unique shard id to number of shard's elements
    shards = {}

    # expected number of elements per shard
    expected_shard_size = total_num_elements // shards_m

    print("Expected shard size: {}".format(expected_shard_size), flush=True)

    # get the seed point, which initially is the first point of the dataset
    points = read_bin(data_file, dtype=dtype, start_idx=0, chunk_size=1)
    num_rows, num_cols = points.shape

    # first seed point
    seed_point_id = 0
    seed_point = points[seed_point_id]
    print("Seed point for shard {}: {}".format(seed_point_id, seed_point), flush=True)
    # remember the seed point id
    processed_point_ids.add(seed_point_id)

    # shard contains the points themselves: we pre-create the numpy array to reuse it multiple times
    shard_points = np.empty((expected_shard_size, num_cols))
    shard_points[0] = seed_point
    running_shard_point_id = 1
    shard_id = 0

    # shard_ids contains the unique point ids as they come in from the input data structure:
    # we pre-create the numpy array to reuse it multiple times
    shard_point_ids = np.empty(expected_shard_size, dtype=np.int32)
    shard_point_ids[0] = shard_id
    shard_saturation_percent = 0

    # all seed points, that are by design cluster centroids
    # these seed points will be stored as a separate HNSW graph
    all_seed_points = [seed_point]
    all_seed_points_ids = [shard_id]

    need_seed_update = False
    is_last_shard_starving = False

    # holds points that do not form a complete shard: we pre-create the numpy array to reuse it multiple times
    special_shard_points = np.empty((expected_shard_size, num_cols))
    special_shard_point_ids = []
    running_special_shard_point_id = 0

    # pre-create the numpy array for a pair of points in multidimensional space
    # the algorithm will reuse this array for computing the distance between the points
    points_pair = np.empty((2, num_cols))

    # TODO number of batches, during which this shard is not growing -- terminate?

    # repeat, while number of shards did not reach the target level M
    while len(shards.keys()) < shards_m:
        # step through the dataset with batch by batch
        for i in range(0, range_upper, BATCH_SIZE):
            # Detailed mem check takes too long time: switched off
            # snapshot = tracemalloc.take_snapshot()
            # display_top(tracemalloc, snapshot)

            print(f"\nProcessing index={i}", flush=True)

            in_loop_points = read_bin(data_file, dtype=np.uint8, start_idx=i, chunk_size=BATCH_SIZE)

            # if last shard was starving, then
            if is_last_shard_starving:
                # re-compute the median distance in this batch, excluding points that were already processed
                points_to_resample = []
                for j in range(0, in_loop_points.shape[0]):
                    candidate_point_id = i + j
                    if candidate_point_id not in processed_point_ids\
                            and \
                            not is_number_in_interval_tree(processed_point_id_interval_tree, candidate_point_id):
                        points_to_resample.append(in_loop_points[j])
                        if len(points_to_resample) == SAMPLE_SIZE:
                            break
                if len(points_to_resample) == SAMPLE_SIZE:
                    computed_dist_max = compute_median_dist(np.array(points_to_resample))
                    print(f"computed {computed_dist_max}", flush=True)
                    print("Updating median distance to this value")
                    if computed_dist_max > dist:
                        dist = computed_dist_max
                    else:
                        # fallback: apply distance multiplier to increase the chances we will make this
                        dist = DIST_MULTIPLIER * dist
                        print(f"Increased the dist to 2x: {dist}", flush=True)
                    # unset the starving shard flat to actually start using this new re-sampled median distance
                    is_last_shard_starving = False
            else:
                print("going inside inner loop by j over current batch of points", flush=True)
                for j in range(0, in_loop_points.shape[0]):
                    # id of the shard candidate is a combination of the running i-th batch and offset j within it
                    candidate_point_id = i + j

                    if candidate_point_id == seed_point_id:
                        print("skipping the original seed point", flush=True)
                        continue

                    if candidate_point_id not in processed_point_ids \
                            and \
                            not is_number_in_interval_tree(processed_point_id_interval_tree, candidate_point_id):
                        # update seed point?
                        if need_seed_update:
                            seed_point = in_loop_points[j]
                            print("Seed point for shard {}: {}".format(i, seed_point))
                            shard_points[0] = seed_point
                            running_shard_point_id = 1
                            all_seed_points.append(seed_point)
                            all_seed_points_ids.append(shard_id)
                            shard_point_ids[0] = i
                            need_seed_update = False
                        else:
                            # seed is up to date and we continue building the shard
                            # dist_j = distance_matrix(np.array([seed_point]), np.array([in_loop_points[j]]))
                            points_pair[0] = seed_point
                            points_pair[1] = in_loop_points[j]
                            if VERBOSE:
                                print(f"points_pair[0]={points_pair[0]}")
                                print(f"points_pair[1]={points_pair[1]}")
                            dist_j = pdist(points_pair)
                            if VERBOSE:
                                print("got dist between seed_point and points[{}]: {}".format(j, dist_j))
                            if dist_j <= dist:
                                if VERBOSE:
                                    print("Got a neighbor!")

                                shard_points[running_shard_point_id,] = in_loop_points[j]
                                shard_point_ids[running_shard_point_id] = candidate_point_id
                                processed_point_ids.add(candidate_point_id)

                                running_shard_point_id += 1

                    # check if we saturated the shard
                    if running_shard_point_id == expected_shard_size:
                        if VERBOSE:
                            print(f"shard_points.shape={shard_points.shape}, shard_point_ids.shape={shard_point_ids.shape}, real size of shard_point_ids={running_shard_point_id}, shard_point_ids={shard_point_ids}")

                        shard = Shard(shard_id, shard_point_ids, shard_points, size=running_shard_point_id)
                        shard_id = add_shard(output_index_path, shard)
                        shards[shard.shardid] = shard.size
                        shard_id += 1
                        need_seed_update = True
                        is_last_shard_starving = False
                        shard_saturation_percent = 0
                        print("Shards built so far: {} with {} keys".format(shards, len(shards.keys())), flush=True)
                        break

                accumulated_points_in_shard = running_shard_point_id
                # if the shard is in point collection phase
                if accumulated_points_in_shard != 0:
                    print("Size of the current shard after going through the current batch: {}".format(accumulated_points_in_shard), flush=True)
                    print("Expected shard size: {}".format(expected_shard_size), flush=True)
                    shard_saturation_percent = (accumulated_points_in_shard / expected_shard_size) * 100
                    print("Saturation {}%".format(shard_saturation_percent))

                # check if we saturated the shard
                if running_shard_point_id == expected_shard_size:
                    shard = Shard(shard_id, shard_point_ids, shard_points, size=running_shard_point_id)
                    shard_id = add_shard(output_index_path, shard)
                    shards[shard.shardid] = shard.size
                    shard_id += 1
                    need_seed_update = True
                    is_last_shard_starving = False
                    shard_saturation_percent = 0
                    print("Shards built so far: {} with {} keys".format(shards, len(shards.keys())), flush=True)

            # release the mem
            if in_loop_points is not None:
                del in_loop_points
                gc.collect()

            # transform set into intervals
            if processed_point_ids is not None:
                intervals = intervals_extract(processed_point_ids)
                processed_point_ids.clear()
                for begin, end in intervals:
                    if begin < end:
                        processed_point_id_interval_tree.append(Interval(begin, end))
                    elif begin == end:
                        processed_point_ids.add(begin)
                processed_point_id_interval_tree.merge_neighbors()

                print(f"size of processed_point_id_interval_tree = {len(processed_point_id_interval_tree)}")
                print(f"size of processed_point_ids = {len(processed_point_ids)}")

        # we reached the end of the whole dataset and can stash existing points into some "special shard"
        if running_shard_point_id < expected_shard_size:
            print("!!! After going through the whole dataset, the shard did not saturate, at size: {} and % = {}".format(running_shard_point_id, shard_saturation_percent), flush=True)
            if shard_saturation_percent > SHARD_SATURATION_PERCENT_MINIMUM:
                shard = Shard(shard_id, shard_point_ids[0:running_shard_point_id], shard_points[0:running_shard_point_id], size=running_shard_point_id)
                shard_id = add_shard(output_index_path, shard)
                shards[shard.shardid] = shard.size
                shard_id += 1
                need_seed_update = True
                is_last_shard_starving = False
                shard_saturation_percent = 0
                print("Shards built so far: {} with {} keys".format(shards, len(shards.keys())), flush=True)
            else:
                # save the current starving shards' points only if we have them ;)
                if running_shard_point_id > 0:
                    # TODO: apply same saturation threshold as for normal shards?
                    for idx, p in enumerate(shard_points):
                        special_shard_points[idx + running_special_shard_point_id,] = shard_points[idx]

                    running_special_shard_point_id = idx + 1

                    special_shard_point_ids.extend(shard_point_ids)
                    print("!!! Appended to the special_shard, its running size: {}".format(len(special_shard_points)), flush=True)
                    special_shard_saturation_percent = (len(special_shard_point_ids) / expected_shard_size) * 100
                    if special_shard_saturation_percent > SHARD_SATURATION_PERCENT_MINIMUM:
                        if running_special_shard_point_id < expected_shard_size:
                            shard = Shard(shard_id, special_shard_point_ids, special_shard_points[0:running_special_shard_point_id], size=running_special_shard_point_id)
                        else:
                            shard = Shard(shard_id, special_shard_point_ids, special_shard_points, size=running_special_shard_point_id)

                        shard_id = add_shard(output_index_path, shard)

                        running_special_shard_point_id = 0
                        shards[shard.shardid] = shard.size
                        shard_id += 1

                        print("Shards built so far: {} with {} keys".format(shards, len(shards.keys())), flush=True)

                need_seed_update = True
                is_last_shard_starving = True

        #snapshot = tracemalloc.take_snapshot()
        #display_top(tracemalloc, snapshot)

    # save the centroid graph
    centroid_shard = Shard(-1, all_seed_points_ids, all_seed_points, size=len(all_seed_points_ids))
    add_shard(output_index_path, centroid_shard)
    print("Saved centroid shard with {} points".format(len(centroid_shard.pointids)))

    print("Processed this many points: {}".format(len(processed_point_ids)), flush=True)


def add_shard(output_index_path, shard):
    """
    Saves shard to disk and returns shard id of the future shard
    """
    print("Saturated shard with id={}. Building HNSW index for it..".format(shard.shardid), flush=True)
    add_points(output_index_path, shard)
    print("Done", flush=True)
    return shard.shardid


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some neighbours.')
    parser.add_argument('--input_file', help='input file with the multidimensional points', required=True)
    parser.add_argument('--output_dir', help='where to store the index', required=True)
    parser.add_argument('-M', type=int, help="expected number of shards, say 1000", required=True)
    parser.add_argument('--dtype', type=str, help="dataset dtype: uint8, float32, int8", required=True)

    args = parser.parse_args()
    print(args)

    points_file = args.input_file
    output_index_path = args.output_dir
    shards_number = args.M
    dtype = args.dtype
    req_type = None

    if dtype == "float32":
        req_dtype = np.float32
        points = read_fbin(points_file, start_idx=0, chunk_size=SAMPLE_SIZE)
    elif dtype == "uint8":
        req_dtype = np.uint8
        points = read_bin(points_file, dtype=req_dtype, start_idx=0, chunk_size=SAMPLE_SIZE)
    else:
        print("Unsupported data type.")
        exit(0)

    computed_dist_max = compute_median_dist(points)
    print(f"computed {computed_dist_max}", flush=True)

    shard_by_dist(points_file, computed_dist_max, output_index_path, dtype=req_dtype, shards_m=shards_number)
