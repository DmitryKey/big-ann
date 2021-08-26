from util.utils import read_fbin, read_bin, get_total_nvecs_fbin, pytorch_cos_sim, ts
from numpy import linalg
from statistics import median
import numpy as np

from torch import stack as torch_stack
from sklearn.cluster import KMeans, MiniBatchKMeans

import nmslib

import sys
import importlib
import pickle

if len(sys.argv)>1:
    config_file = sys.argv[1]
else:
    config_file = 'config_small'
config = importlib.import_module(config_file)

#See config.small.py for the config options descriptions
RANDOM_SEED = config.RANDOM_SEED
SAMPLE_SIZE = config.SAMPLE_SIZE
BATCH_SIZE = config.BATCH_SIZE
MAX_POINTS = config.MAX_POINTS
S = config.S

#Renders the filename for a shard
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
Adds a batch of points to a specific shard
"""
def add_points(path,name,ids,points):
    shardpath = shard_filename(path,name)
    shard = nmslib.init(method='hnsw', space='l2')
    shard.loadIndex(shardpath,load_data=True)
    shard.addDataPointBatch(points,ids)
    shard.createIndex(print_progress=False)
    shard.saveIndex(shardpath,save_data=True)

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
    
    print(f'Loading KMeans: {ts()}')
    kmeans = pickle.load(open(centroids_filename(path), "rb"))
    centroids = kmeans.cluster_centers_
    show_distance_stats(centroids)

    print(f'Creating Shards: {ts()}')
    for i in range(len(centroids)):
        add_shard(path,i)

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

    #Load and index the datafile in batches
    for batch in range(0, range_upper, batch_size):

        print(f"Processing index {batch}: {ts()}")
        points = read_bin(data_file, dtype, start_idx=batch, chunk_size=batch_size)

        #get the centroids for all the points in the batch
        results = kmeans.predict(points)

        #group the points by centroid
        group_ids = {}
        group_points = {}
        distances = []
        for i in range(len(points)):
            point_id = batch+i
            point = points[i] #the point vector
            key = results[i] #index of the centroid
            centroid = centroids[key]
            distance = np.linalg.norm(centroid-points) #l2 distance
            distances.append(distance)
            if key not in group_ids:
                group_ids[key] = []
                group_points[key] = []
            group_ids[key].append(point_id)
            group_points[key].append(point)

        med = median(distances)
        medians.append(med)
        print(f' Median: {med}')

        #add the points to the appropriate shards
        for key in group_ids.keys():
            add_points(path,key,group_ids[key],group_points[key])

        #assert len(list(group_ids.keys())) == len(points)

    print(f"Done! {ts()}")

index_dataset("../data/shards/","../data/bigann/learn.100M.u8bin",np.uint8)

"""
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
"""