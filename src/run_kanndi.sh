# fix the util package reading issue
export PYTHONPATH=.
# 100M points
#python algorithms/sharding/kanndi/shard_by_distance.py --input_file /datadrive/big-ann-benchmarks/data/bigann.bak/base.1B.u8bin.crop_nb_100000000 --output_dir /datadrive/big-ann/data -M 100
# 1B points
python algorithms/sharding/kanndi/shard_by_distance.py --input_file /datadrive/big-ann-benchmarks/data/bigann/base.1B.u8bin --output_dir /datadrive/big-ann/data.1B/ -M 1000
