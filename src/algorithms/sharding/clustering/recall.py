from util.utils import read_fbin, read_bin, write_bin, get_total_nvecs_fbin, pytorch_cos_sim, ts
from numpy import linalg
from statistics import median
import numpy as np

from torch import stack as torch_stack
from sklearn.cluster import KMeans, MiniBatchKMeans

import os
import sys
import importlib
import json
import pickle

if len(sys.argv)>1:
    config_file = sys.argv[1]
else:
    config_file = 'config_bigann_small'
config = importlib.import_module(config_file)

#Where's the data
INDEX_PATH = config.INDEX_PATH
DATA_TYPE = config.DATA_TYPE
DATA_FILE = config.DATA_FILE
QUERY_FILE = config.QUERY_FILE

#See config.small.py for the config options descriptions
BATCH_SIZE = config.BATCH_SIZE

#Renders the filename for the kmeans pickle
def centroids_filename(path):
    return f'{path}centroids_{config_file}.pickle'

def get_exact(queries,points):
    results = []
    #this should be vectorized but it's saturday and I'm done fighting with numpy 😂
    for i,q in enumerate(queries):
        best = -1
        bestid = -1
        for j,p in enumerate(points):
            dist = np.linalg.norm(q-p)
            if best == -1 or dist<best:
                best = dist
                bestid = j
        results.append(bestid)
    return results

"""
Creates the index and shard graphs for an entire dataset
"""
def test_kmeans_recall(
        centroid_file,
        query_file,
        dtype,
        batch_size: int = BATCH_SIZE
    ):

    print(f"Loading queries from {query_file}: {ts()}")
    queries = read_bin(query_file, dtype, start_idx=0, chunk_size=10000)

    print(f'Loading KMeans from {centroid_file}: {ts()}')
    kmeans = pickle.load(open(centroid_file, "rb"))
    centroids = kmeans.cluster_centers_

    print(f'exact: {ts()}')
    exact = get_exact(queries,centroids)

    print(f'predictions: {ts()}')
    predictions = kmeans.predict(queries)

    print(f"Done! {ts()}")

    assert len(exact) == len(predictions)
    tp = 0
    fn = 0
    for i in range(len(exact)):
        if exact[i]==predictions[i]:
            tp += 1
        else:
            fn += 1
    recall = tp/(tp+fn)

    print(f"Recall: {recall}")


if __name__ == "__main__":
    test_kmeans_recall(centroids_filename(INDEX_PATH),QUERY_FILE,DATA_TYPE)
    