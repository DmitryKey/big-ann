import numpy as np
from sklearn.preprocessing import normalize
import struct
import torch
from torch import Tensor
import datetime
import math
import nmslib
import hnswlib
import linecache
import os
import gc
import itertools
from intervaltree import Interval, IntervalTree

VERBOSE = False


def ts():
    """
    Gets an ISO string timestamp, helps with seeing how long things took to run
    """
    return str(datetime.datetime.now())


def get_solr_vector_search(bert_client, query):
    """
    Takes user keyword query, computes BERT embedding and returns a
    comma-separated string with vector points
    :param bert_client: used for computing BERT embedding
    :param query: user keyword query
    :return: CSV string suitable for querying in Solr
    """
    return ','.join(str(elem) for elem in bert_client.encode([query]).flat)


def to_solr_vector(vectors):
    """
    Takes BERT vectors array and converts into indexed representation: like so:
    1|point_1 2|point_2 ... n|point_n
    :param vectors: BERT vector points
    :return: Solr-friendly indexed representation targeted for indexing
    """
    solr_vector = []
    for vector in vectors:
        for i, point in enumerate(vector):
            solr_vector.append(str(i) + '|' + str(point))
    solr_vector = " ".join(solr_vector)
    return solr_vector


def myfmt(r):
    return "%.10f" % (r,)


def get_elasticsearch_vector(query_vector):
    """
    Compute the BERT embedding of the given query and return an array of vector values
    :type query_vector: embedding vector of the query
    :return: BERT embedding array
    """
    # 1.
    #vecfmt = np.vectorize(myfmt)
    #return vecfmt(query_vector)

    # 2.
    query_vector = normalize(query_vector, norm='l2', axis=1)
    query_vector = np.round(query_vector.astype(np.float64), 10)
    return query_vector.flatten().tolist()


"""
                  IO Utils for reading and writing binary vectors
"""


def get_total_nvecs_fbin(filename):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)

    return nvecs


def get_total_dim_fbin(filename):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)

    return dim


def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
        if arr.size > 0:
            return arr.reshape(nvecs, dim)
        else:
            return np.zeros(shape=(1, dim))


# by Leo Joffe
def read_bin(filename, dtype, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        # The header is two np.int32 values
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size

        if dtype == np.uint8:
            type_multiplier = 1
        elif dtype == np.float32:
            type_multiplier = 4

        arr = np.fromfile(f, count=nvecs * dim, dtype=dtype, offset=start_idx * dim * type_multiplier)
    # Reshaping an array may or may not involve a copy. The reasons will be explained in the How it works... section.
    # For instance, reshaping a 2D matrix does not involve a copy, unless it is transposed
    # (or more generally, non-contiguous):
    # Source: https://ipython-books.github.io/45-understanding-the-internals-of-numpy-to-avoid-unnecessary-array-copying
    # return arr.reshape(-1, dim)
    return arr.T.reshape(-1, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32,
                          offset=start_idx * dim)
    return arr.reshape(nvecs, dim)


def write_fbin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:s
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('float32').flatten().tofile(f)


def write_ibin(filename, vecs):
    """ Write an array of int32 vectors to *.ibin file
    Args:
        :param filename (str): path to *.ibin file
        :param vecs (numpy.ndarray): array of int32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('int32').flatten().tofile(f)


def fbin_to_tsv(bin_fname: str, tsv_fname: str, total_elems: int):
    arr = read_fbin(bin_fname, chunk_size=total_elems)
    print(np.shape(arr))
    np.savetxt(tsv_fname, arr, delimiter="\t")


# by Max Irwin
def write_bin(filename, dtype, vecs):
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        print(vecs.shape)
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype(dtype).flatten().tofile(f)

def buddy_up(points,friends):
    #rearranges the points to put buddy dimensions next to each other.
    #this will be used during PQ to get ideal subvectors without needing to modify Faiss
    #friends is a 1D array of dimension indexes in order of their buddiness.
    #for example if dims 12,17,18,50,52,96,101,113 are buddies, then they would appear as:
    #[...,12,17,18,50,52,96,101,113,...] in the friends param.
    #This is a beautiful one-liner in numpy...
    return points[:,friends]

"""
import mmap
def mmap_bin(filename, dtype):
    with open(filename, mode="wb") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_WRITE) as mmap_obj:
            text = mmap_obj.read()
            print(text)
"""


# by UKPLab
# From https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py (MIT License)
def pytorch_cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def entropy(labels, base=None):
    """
    https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python#45091961
    """
    value,counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = math.e if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()


def shard_filename(path,name):
    """
    Renders the filename for a shard
    """
    return f'{path}shard{name}.hnsw'


class Shard:
    def __init__(self, shard_id: int, point_ids: np.array, points: np.array, size: int, shard_saturation_percent: float, dim: int):
        self.shardid = shard_id
        self.pointids = point_ids
        self.points = points
        self.size = size
        self.shard_saturation_percent = shard_saturation_percent
        self.dim = dim


class SpacePoint:
    def __init__(self, id: int, point: np.array):
        self.point_id = id
        self.point = point


def add_points(path, shard: Shard):
    """
    Adds a batch of points to a specific shard
    """
    shardpath = shard_filename(path, shard.shardid)
    index = nmslib.init(method='hnsw', space='l2')
    if VERBOSE:
        print(f"add_points(): type(shard.pointids)={type(shard.pointids)}")
    index.addDataPointBatch(shard.points, ids=shard.pointids)
    index.createIndex(print_progress=False)
    index.saveIndex(shardpath, save_data=True)
    del index
    gc.collect()


def save_shard(path, shard: Shard):
    """
    Adds a batch of points to a specific shard
    """
    shardpath = shard_filename(path, shard.shardid)
    p = hnswlib.Index(space='l2', dim=shard.dim)
    p.init_index(max_elements=shard.size, ef_construction=200, M=16)
    p.add_items(shard.points, ids=shard.pointids)
    p.set_ef(50)
    p.save_index(shardpath)
    del p


# Loads index from disk
def load_index(filename):
    index = nmslib.init(method='hnsw', space='l2')
    index.createIndex(print_progress=True)
    index.loadIndex(filename)
    return index


# Searches the given shard
def query_shard(shard_name, query):
    shard = nmslib.init(method='hnsw', space='l2')
    shard.loadIndex(shard_name, load_data=True)
    results, distances = shard.knnQuery(query, k=10)
    return results, distances


# RAM consumption monitoring
# credit: https://stackoverflow.com/a/45679009/158328
def display_top(tracemalloc, snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def intervals_extract(iterable: set):
    """
    The other pythonic method is to use Python itertools. We use itertools.groupby().
    Where enumerate(iterable) is taken as iterable and lambda t: t[1] – t[0]) as key function
    to find the sequence for intervals.

    Source: https://www.geeksforgeeks.org/python-make-a-list-of-intervals-with-sequential-numbers/
    """
    iterable = sorted(iterable)
    #print(f"intervals_extract() iterable={iterable}")
    intervals = list()
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        intervals.append([group[0][1], group[-1][1]])

    #print(f"intervals={intervals}")
    return intervals


def is_number_in_intervals(intervals: list, target: int):
    """
    Traverses intervals and checks if the given number is inside an interval.
    Example:
        interval: [[1, 6], [45, 48], [110, 112]]
        number: 2
        return: True
    """
    for interval in intervals:
        # print(f"interval={interval} type(interval)={type(interval)} target={target} type(target)={type(target)}")
        if interval[0] <= target <= interval[1]:
            return True
    return False


def append_intervals_to_tree(intervals: list, tree: IntervalTree):
    tree.update(
        Interval(begin, end) for begin, end in intervals
    )
    tree.merge_neighbors()

    return tree


def is_number_in_interval_tree(t: IntervalTree, target: int):
    if t is None:
        return False

    if len(t[target]) > 0:
        return True
    return False
