import argparse
import datetime
import numpy as np

# Total number of queries to search for (sequential)
from util.utils import load_index, read_fbin, read_bin, query_shard

MAXIMUM_QUERIES = 10000

MAX_CENTROIDS = 3


# Gets an ISO string timestamp, helps with seeing how long things took to run
def ts():
    return str(datetime.datetime.now())


# Renders the filename of the centroids (seed points, remember?) shard
def centroids_filename(path):
    return f'{path}shard-1.hnsw'


# Renders the filename for a shard
def shard_filename(path, name):
    return f'{path}shard{name}.hnsw'


def query_index(path, query_file, dtype, k=10):
    # Get the centroid index
    print(f'Load Centroid Index: {ts()}')
    centroid_index = load_index(centroids_filename(path))
    start_time = datetime.datetime.now().timestamp()
    print(f'Search Centroid Index for {MAXIMUM_QUERIES} queries: {ts()}')

    if dtype == "float32":
        req_dtype = np.float32
        query_points = read_fbin(query_file, start_idx=0, chunk_size=MAXIMUM_QUERIES)
    elif dtype == "uint8":
        req_dtype = np.uint8
        query_points = read_bin(query_file, dtype=req_dtype, start_idx=0, chunk_size=MAXIMUM_QUERIES)
    else:
        print("Unsupported data type.")
        exit(0)

    qnum = 0
    for query in query_points:
        # get the centroids for the query
        centroids, centroid_distances = centroid_index.knnQuery(query, k=MAX_CENTROIDS)

        # search the shard
        shard_name = shard_filename(path, centroids[0])
        results, result_distances = query_shard(shard_name, query)

        # log results
        print(f'\nFound {query} in shard {centroids[0]}: '
              f'closest point: {results[0]} with the distance of {result_distances[0]} at {ts()}')
        print('All results for this query')
        for i in range(len(results)):
            print(f'{qnum} result {i} :: {result_distances[i]} {results[i]}')
        qnum += 1
        print('--------------------------------')

    print(f"Done! {ts()}")
    end_time = datetime.datetime.now().timestamp()
    seconds = end_time - start_time
    print(f"Queries Per Second: {MAXIMUM_QUERIES / seconds}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search some KANNDI neighbours.')
    parser.add_argument('--query_file', help='input file with the multidimensional query points', required=True)
    parser.add_argument('--index_dir', help='index directory with shards storing multidimensional points', required=True)
    parser.add_argument('--dtype', type=str, help="query and index dataset dtype: uint8, float32, int8", required=True)

    args = parser.parse_args()
    print(args)

    query_index(args.index_dir, args.query_file, args.dtype)