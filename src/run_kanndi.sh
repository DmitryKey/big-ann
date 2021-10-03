# fix the util package reading issue
export PYTHONPATH=.
python algorithms/sharding/kanndi/shard_by_distance.py /datadrive/big-ann-benchmarks/data/bigann.bak/base.1B.u8bin.crop_nb_100000000  /datadrive/big-ann/data/
