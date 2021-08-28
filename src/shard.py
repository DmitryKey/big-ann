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
import tqdm

if len(sys.argv)>1:
    config_file = sys.argv[1]
else:
    config_file = 'config_small'
config = importlib.import_module(config_file)

#Where's the data
INDEX_PATH = config.INDEX_PATH
DATA_TYPE = config.DATA_TYPE
DATA_FILE = config.DATA_FILE
QUERY_FILE = config.QUERY_FILE

#See config.small.py for the config options descriptions
RANDOM_SEED = config.RANDOM_SEED
SAMPLE_SIZE = config.SAMPLE_SIZE
BATCH_SIZE = config.BATCH_SIZE
MAX_POINTS = config.MAX_POINTS
S = config.S

#Renders the filename for a shard bucket
def bucket_filename(path,name):
    return f'{path}bucket{name}.u8bin',f'{path}bucket{name}.json',

#Renders the filename for a shard graph
def shard_filename(path,name):
    return f'{path}shard{name}.hnsw'

#Renders the filename for the kmeans pickle
def centroids_filename(path):
    return f'{path}centroids_{config_file}.pickle'

#Show the extremes of the similarity scores between all the centroids
def show_distance_stats(points):
    similarities = pytorch_cos_sim(points,points)
    scores = []
    for a in range(len(similarities)-1):
        for b in range(a+1, len(similarities)):
            scores.append(float(similarities[a][b]))
    scores = sorted(scores)
    print(f'Farthest:{scores[0]}    Median:{median(scores)}     Closest:{scores[len(scores)-1]}')


"""
Creates a new shard graph for a centroid shard
The shard is an HNSW graph with neighborhoods of the parent centroid.
The shard is persisted to disk for each addition.
The shard is loaded from disk and searched when a query is in its centroid neighborhood.
"""
def add_shard(path,name):
    shard = nmslib.init(method='hnsw', space='l2')
    shard.createIndex(print_progress=False)
    shard.saveIndex(shard_filename(path,name),save_data=True)


"""
Creates the index and shard graphs for an entire dataset
"""
def index_dataset(
        path,
        data_file, 
        dtype, 
        batch_size: int = BATCH_SIZE, 
        sample_size: int = SAMPLE_SIZE, 
        n_clusters: int = S, 
        max_points: int = MAX_POINTS
    ):
    
    print(f'Loading KMeans from {centroids_filename(path)}: {ts()}')
    kmeans = pickle.load(open(centroids_filename(path), "rb"))
    centroids = kmeans.cluster_centers_
    #show_distance_stats(centroids)

    #print(f'Creating Buckets in {path}: {ts()}')
    #for i in range(len(centroids)):
    #    add_shard(path,i)

    #Prepare for batch indexing
    total_num_elements = get_total_nvecs_fbin(data_file)
    if max_points and max_points<total_num_elements:
        range_upper = max_points
    else:
        range_upper = total_num_elements

    print(f"Total number of points in dataset: {total_num_elements}")
    print(f"Maximum number of points to index: {range_upper}")
    print(f"Reading data from {data_file} in {BATCH_SIZE} chunks")

    #median distances from centroids, to track drift from the sample
    medians = []

    #group the points by centroid
    groups = {}

    #Organize ids and distances by centroid:
    for batch in range(0, range_upper, batch_size):

        print(f"Predicting points {batch} to {batch+batch_size}: {ts()}")
        points = read_bin(data_file, dtype, start_idx=batch, chunk_size=batch_size)

        #get the centroids for all the points in the batch
        results = kmeans.predict(points)

        print(f"Organizing shard: {ts()}")

        distances = []
        for i in tqdm.tqdm(range(len(points))):
            point_id = batch+i
            point = points[i] #the point vector
            key = results[i] #index of the centroid
            centroid = centroids[key]
            distance = np.linalg.norm(centroid-point) #l2 distance
            distances.append(distance)
            if key not in groups:
                groups[key] = []
            groups[key].append({"id":point_id,"distance":distance})
                
        med = median(distances)
        medians.append(med)
        print(f' Median: {med}')

    
    #Split into buckets on disk
    for key in sorted(groups.keys()):
        group = groups[key]
        print(f"Saving bucket {key}: {ts()}")
        bucket = np.empty((0,128))
        for batch in range(0, range_upper, batch_size):
            points = read_bin(data_file, dtype, start_idx=batch, chunk_size=batch_size)
            head = batch
            tail = batch + batch_size
            for row in group:
                if head<=row['id'] or row['id']>tail:
                    bucket = np.vstack([bucket,point])

        bucketpath,jsonpath = bucket_filename(path,key)
        write_bin(bucketpath,DATA_TYPE,bucket)
        with open(jsonpath, "w") as f:
            f.write(json.dumps(group))

    print(f"Done! {ts()}")

"""
Creates the index and shard graphs for an entire dataset
"""
def speed_read_test(
        path,
        data_file, 
        dtype, 
        batch_size: int = BATCH_SIZE, 
        sample_size: int = SAMPLE_SIZE, 
        n_clusters: int = S, 
        max_points: int = MAX_POINTS
    ):

    #Prepare for batch indexing
    total_num_elements = get_total_nvecs_fbin(data_file)
    if max_points and max_points<total_num_elements:
        range_upper = max_points
    else:
        range_upper = total_num_elements

    print(f"Total number of points in dataset: {total_num_elements}")
    print(f"Maximum number of points to index: {range_upper}")
    print(f"Reading data from {data_file} in {BATCH_SIZE} chunks")

    #Load and index the datafile in batches
    for batch in range(0, range_upper, batch_size):
        print(f"Reading points {batch} to {batch+batch_size}: {ts()}")
        points = read_bin(data_file, dtype, start_idx=batch, chunk_size=batch_size)


if __name__ == "__main__":
    index_dataset(INDEX_PATH,DATA_FILE,DATA_TYPE)
    #speed_read_test(INDEX_PATH,DATA_FILE,DATA_TYPE)