from util.utils import read_bin, get_total_nvecs_fbin, pytorch_cos_sim, ts
from statistics import median
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans

import sys
import importlib
import pickle

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = 'config_small'
config = importlib.import_module(config_file)

# Where's the data
INDEX_PATH = config.INDEX_PATH
DATA_TYPE = config.DATA_TYPE
DATA_FILE = config.DATA_FILE
QUERY_FILE = config.QUERY_FILE

# See config.small.py for the config options descriptions
RANDOM_SEED = config.RANDOM_SEED
SAMPLE_SIZE = config.SAMPLE_SIZE
BATCH_SIZE = config.BATCH_SIZE
MAX_ITER = config.MAX_ITER
S = config.S


def centroids_filename(path):
    """
    Renders the filename for the kmeans pickle
    """
    return f'{path}centroids_{config_file}.pickle'


def show_distance_stats(allpoints):
    """
    Show the extremes of the similarity scores between all the centroids
    """
    #points = np.random.choice(allpoints,size=100)
    points = allpoints[np.random.choice(allpoints.shape[0], size=min(len(allpoints),500), replace=False)]
    similarities = pytorch_cos_sim(points,points)
    scores = []
    for a in range(len(similarities)-1):
        for b in range(a+1, len(similarities)):
            scores.append(float(similarities[a][b]))
    scores = sorted(scores)
    print(f'  Farthest:{scores[0]}    Median:{median(scores)}     Closest:{scores[len(scores)-1]}')


def find_centroids(data_file, dtype, sample_size: int = SAMPLE_SIZE, n_clusters: int = S, max_iter: int = MAX_ITER):
    """
    This will take a sample of the dataset to fit centroids that will be used as shard entry points
    """
    print(f'Loading Samples: {ts()}')
    points = read_bin(data_file, dtype, start_idx=0, chunk_size=sample_size)
    print(f'Clustering dataset shape: {points.shape}')
    print(f'Starting KMeans: {ts()}')
    if RANDOM_SEED:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, max_iter=max_iter, verbose=1).fit(points)
    else:
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, verbose=1).fit(points)
    
    return kmeans.cluster_centers_


def find_centroids_batch(
        path, 
        data_file, 
        dtype,
        sample_size: int = SAMPLE_SIZE, 
        batch_size: int = BATCH_SIZE,
        n_clusters: int = S, 
        max_iter: int = MAX_ITER
    ):
    """
    This will minibatch on a sample of the dataset to fit centroids that will be used as shard entry points
    """

    # Prepare for batch indexing
    total_num_elements = get_total_nvecs_fbin(data_file)
    if sample_size and sample_size<total_num_elements:
        range_upper = sample_size
    else:
        range_upper = total_num_elements

    print(f"{data_file} sample_size={sample_size} batch_size={batch_size} n_clusters={n_clusters} max_iter={max_iter}")
    print(f"Total number of points in dataset: {total_num_elements}")
    print(f"Maximum number of points to index: {range_upper}")
    print(f'Starting KMeans: {ts()}')
    if RANDOM_SEED:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, max_iter=max_iter, verbose=1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter, batch_size=batch_size,verbose=1)
    
    # Load and index the datafile in batches
    for batch in range(0, range_upper, batch_size):

        points = read_bin(data_file, dtype, start_idx=batch, chunk_size=batch_size)
        print(f"Processing kmeans {batch} {str(points.shape)}: {ts()}")
        kmeans = kmeans.partial_fit(points)
        show_distance_stats(kmeans.cluster_centers_)

    pickle.dump(kmeans, open(centroids_filename(path), "wb"))
    if config_file == 'config_small':
        kmeans_test = pickle.load(open(centroids_filename(path), "rb"))
        print(kmeans_test.cluster_centers_)

    return kmeans


if __name__ == "__main__":
    #find_centroids_batch("../data/shards/","../data/bigann/learn.100M.u8bin",np.uint8)
    find_centroids_batch(INDEX_PATH,DATA_FILE,DATA_TYPE)
    print(f"Done! {ts()}")