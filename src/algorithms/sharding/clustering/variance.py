import sys
sys.path.insert(1, '../../../')

import numpy as np
import pandas as pd
import math

from util.utils import read_fbin, read_bin, get_total_nvecs_fbin, get_total_dim_fbin, pytorch_cos_sim, ts, entropy
from numpy import linalg
from statistics import median
from scipy.stats import anderson

from torch import stack as torch_stack

import importlib
import pickle


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
MAX_ITER = config.MAX_ITER
S = config.S

"""
This will get the variance and entropy for dimensions of a dataset
"""
def calculate_variance(
        path, 
        data_file, 
        dtype,
        sample_size: int = SAMPLE_SIZE, 
        batch_size: int = BATCH_SIZE,
        n_clusters: int = S, 
        max_iter: int = MAX_ITER
    ):

    #Prepare for batch indexing
    total_num_elements = get_total_nvecs_fbin(data_file)
    total_num_dimensions = get_total_dim_fbin(data_file)
    if sample_size and sample_size<total_num_elements:
        range_upper = sample_size
    else:
        range_upper = total_num_elements

    print(f"{data_file} sample_size={sample_size} batch_size={batch_size} n_clusters={n_clusters} max_iter={max_iter}")
    print(f"Total number of dimensions in dataset: {total_num_dimensions}")
    print(f"Total number of points in dataset: {total_num_elements}")
    
    print(f"Maximum number of points to index: {range_upper}")

    dims = []
    variance = []
    entropies = []
    covariance_num = total_num_dimensions-1

    assert(sample_size<=100000)
    
    points = read_bin(data_file, dtype, start_idx=0, chunk_size=sample_size)
    sample_variance = np.var(points)

    #Make a dataframe
    schema = {"dimension":[],"variance":[],"entropy":[]}
    for codim in range(covariance_num):
        schema[f'codimension_{codim}'] = []
        schema[f'covariance_{codim}'] = []
    df = pd.DataFrame(schema)

    #Load and index the datafile in batches
    for dim in range(total_num_dimensions):

        #Scalar values of a specific dimension for all points
        dim_points = points[:,dim]

        v = np.var(dim_points)
        e = entropy(dim_points)
        a = anderson(dim_points,dist='norm')
        print(a)

        #Find the lowest covariance dim:
        covars = []
        min_covar_val = sample_variance
        min_covar_dim = 999

        #Compare with every other dimension's variance:
        for dim2 in range(total_num_dimensions):
            if dim==dim2:
                continue
            dim2_points = points[:,dim2]
            covar_points = np.concatenate((dim_points,dim2_points))
            covar = np.var(covar_points)
            covars.append((covar,dim2))
        
        #which dimensions covariance is best?
        covars = sorted(covars, key = lambda x: x[0])

        row = {"dimension":dim,"variance":v,"entropy":e}
        for codim in range(covariance_num):
            row[f'codimension_{codim}'] = covars[codim][1]
            row[f'covariance_{codim}'] = covars[codim][0]
        df = df.append(row,ignore_index=True)

        variance.append(v)
        entropies.append(e)

    print(df)

    df.to_csv(f'variance_{config_file}.csv')

    variance_sorted = sorted(variance)
    print(variance_sorted)

    variance2 = np.asarray(variance)
    print(np.var(points),np.var(variance2),entropy(variance2))

if __name__ == "__main__":
    calculate_variance(INDEX_PATH,DATA_FILE,DATA_TYPE)
    print(f"Done! {ts()}")