import sys

from util.utils import read_fbin, read_bin, get_total_nvecs_fbin, get_total_dim_fbin, add_points
from numpy import linalg
from statistics import median
import numpy as np

# desired number of shardsCreates a new shard graph for a centroid shard
# The shard is an HNSW graph with neighborhoods of the parent centroid.
# The shard is persisted to disk for each addition.
# The shard is loaded from disk and searched when a query is in its centroid neighborhood.
M = 1000

# target maximum distance between points to fall inside a shard
DIST_MAX = 100

# number of times to sample the input dataset to approximate the dist_max
# SAMPLE_TIMES = 10
# size of the sample of points examined linearly during max dist computation
SAMPLE_SIZE = 1000

# batch size for reading points from the input file during the sharding algorithm
BATCH_SIZE = 1000000


# expected: 1 280 000 008
# file size: 1 280 0 000 008
def compute_max_dist(data_file: str, sample_size: int = SAMPLE_SIZE)->float:
    points = read_fbin(data_file, start_idx=0, chunk_size=sample_size)
    #points = read_bin(filename=data_file, dtype=np.float32, start_idx=0, chunk_size=sample_size)
    print(points.shape)
    num_rows, num_cols = points.shape
    print(num_rows)

    dists = np.sqrt(np.sum((points[None, :] - points[:, None])**2, -1))
    # dists = []
    # for i in range(0,num_rows):
    #    for j in range(1,num_rows-1):
    #        dist = linalg.norm(points[i]-points[j])
    #        dists.append(dist)

    print(dists)

    return median(dists.flatten())


def shard_by_dist(data_file: str, dist: float, output_index_path: str, shards_m: int = M):
    # set of integer order ids of each point that was already placed into a shard => processed
    processed_point_ids = set()
    complete_shards = 0

    total_num_elements = get_total_nvecs_fbin(data_file)
    dimensionality = get_total_dim_fbin(data_file)
    print(f"Total number of points to process: {total_num_elements}")
    print(f"Reading data from {data_file} in {BATCH_SIZE} chunks")

    range_upper = total_num_elements
    print(f"range_upper={range_upper}")

    # map from shard id to number of elements
    shards = {}

    # expected number of elements per shard
    expected_shard_size = total_num_elements / shards_m

    # repeat, while number of shards did not reach the target level M
    while complete_shards < shards_m:
        complete_shards = len(shards.keys())
        #shard = np.zeros(shape=(shards_m, dimensionality))
        shard = []
        shard_ids = []
        # step through the dataset with BATCH_SIZE window
        for i in range(0, range_upper, BATCH_SIZE):
            print(f"Processing index={i}")
            points = read_fbin(data_file, start_idx=i, chunk_size=BATCH_SIZE)
            # print(points.shape)
            # print(points.shape[0])

            # Proceed with processing the batch, if this point has not been visited previously.
            # TODO: Picking the first point in each batch as the starting point for sharding might be suboptimal.
            if i not in processed_point_ids:
                # fix the starting point
                first_point = points[0]
                # mark it visited
                processed_point_ids.add(i)
                shard.append(first_point)
                shard_ids.append(i)
                ## drop it from the input points
                # points = np.delete(points, first_point, axis=0)
                print("going inside inner loop by j over current batch of points, skipping the first point")
                for j in range(1, points.shape[0]):
                    # id of the candidate is a combination of the running i-th batch and offset j within it
                    candidate_point_id = i + j
                    if candidate_point_id not in processed_point_ids:
                        dist_j = linalg.norm(first_point-points[j])
                        if dist_j <= dist:
                            processed_point_ids.add(candidate_point_id)
                            shard.append(points[j])
                            shard_ids.append(candidate_point_id)

            # check if we saturated the shard
            if len(shard) == expected_shard_size:
                add_points(output_index_path, "shard_" + str(i), shard_ids, shard)

    print("Processed points: {}".format(len(processed_point_ids)))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Expect two params: (1) an input points file (2) an output index path")
        exit(0)

    points_file = sys.argv[1]
    output_index_path = sys.argv[2]

    computed_dist_max = compute_max_dist(points_file)
    print(f"computed {computed_dist_max}")

    shard_by_dist(points_file, computed_dist_max, output_index_path)
