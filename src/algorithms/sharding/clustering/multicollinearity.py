import sys
sys.path.insert(1, '../../../')

import numpy as np
import pandas as pd
import math

import seaborn as sns
sns.set_theme()

from util.utils import read_fbin, read_bin, get_total_nvecs_fbin, get_total_dim_fbin, pytorch_cos_sim, ts, entropy
from numpy import linalg
from statistics import median
from scipy.stats import anderson,kstest

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
from scipy import interpolate
import numpy as np
def bimodal_split_point(hist)
    t=np.linspace(0,1,200)
    x=np.cos(5*t)
    y=np.sin(7*t)
    tck, u = interpolate.splprep([x,y])

    ti = np.linspace(0, 1, 200)
    dxdt, dydt = interpolate.splev(ti,tck,der=1)
"""

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

    #just a safety precaution.  These tests can get big!  Remove at your own risk
    assert(sample_size<=100000)
    
    #Read all the points of the sample_size into memory
    points = read_bin(data_file, dtype, start_idx=0, chunk_size=sample_size)
    
    df = pd.DataFrame(points)
    print(df.shape)

    cor = df.corr()
    inv = np.linalg.inv(cor.values)
    vif = pd.DataFrame(inv, index = cor.index, columns=cor.columns)
    vif.to_csv(f'multicollinearity_{config_file}.csv')
    
    vals = {}
    for i in range(vif.shape[0]):
        for j in range(vif.shape[1]):
            pair = f'{min(i,j)}_{max(i,j)}'
            if i==j or pair in vals.keys():
                continue
            val = vif.iloc[i,j]
            vals[pair] = val
            if val>1.3:
                print(i,j,val)
    sorted_vals = sorted(vals.items(), reverse=True, key=lambda item: item[1])    
    print(sorted_vals[:100])
    
    scale = 2
    wd = 11.7 * scale
    ht = 8.27 * scale
    vmax = sorted_vals[0][1]
    vmin = vmax * -1
    sns.set(rc={'figure.figsize':(wd,ht)})

    #heatmap of VIF
    heat = sns.heatmap(inv,annot=False,center=0,vmax=vmax,vmin=vmin,square=True)
    fig = heat.get_figure()
    fig.savefig(f'multicollinearity_heatmap_{config_file}.png')

    #Only show half
    for i in range(0,vif.shape[0]):
        for j in range(i,vif.shape[1]):
            vif.iloc[i,j] = 0.0
    heat2 = sns.heatmap(inv,annot=False,center=0,vmax=vmax,vmin=vmin,square=True)
    fig2 = heat2.get_figure()
    fig2.savefig(f'multicollinearity_heatmap2_{config_file}.png')


if __name__ == "__main__":
    calculate_variance(INDEX_PATH,DATA_FILE,DATA_TYPE)
    print(f"Done! {ts()}")