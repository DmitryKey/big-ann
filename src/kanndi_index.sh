# fix the util package reading issue
export PYTHONPATH=.
###
# BIGANN
###

# !!! LOCAL !!!
# 100M points
# CPU profiling
#python -m cProfile -o /Users/dmitry/Desktop/BigANN/datasets/bigann/program.prof algorithms/sharding/kanndi/shard_by_distance.py --input_file /Users/dmitry/Desktop/BigANN/datasets/bigann/learn.100M.u8bin --output_dir /Users/dmitry/Desktop/BigANN/datasets/bigann/data.100M/ -M 100 --dtype uint8
# RAM profiling
# python -m memory_profiler algorithms/sharding/kanndi/shard_by_distance.py --input_file /Users/dmitry/Desktop/BigANN/datasets/bigann/learn.100M.u8bin --output_dir /Users/dmitry/Desktop/BigANN/datasets/bigann/data.100M/ -M 100 --dtype uint8
# 10M points
python -m memory_profiler algorithms/sharding/kanndi/shard_by_distance.py --input_file /Users/dmitry/Desktop/BigANN/datasets/bigann/base.1B.u8bin.crop_nb_10000000 --output_dir /Users/dmitry/Desktop/BigANN/datasets/bigann/data.10M/ -M 10 --dtype uint8

# !!! SERVER !!!
# 10M points
# python -m memory_profiler algorithms/sharding/kanndi/shard_by_distance.py --input_file /datadrive/big-ann-benchmarks/data/bigann/base.1B.u8bin.crop_nb_10000000 --output_dir /datadrive/big-ann/index/bigann/data.10M/ -M 10 --dtype uint8

# 1B points
# python algorithms/sharding/kanndi/shard_by_distance.py --input_file /datadrive/big-ann-benchmarks/data/bigann/base.1Billion.u8bin --output_dir /datadrive/big-ann/index/bigann/data.1B/ -M 1000 --dtype uint8


###
# Text2Image
###
#python algorithms/sharding/kanndi/shard_by_distance.py --input_file /datadrive/big-ann-benchmarks/data/text2image1B/base.1B.fbin.crop_nb_100000000 --output_dir /datadrive/big-ann/text2image/data.1B/ -M 100

###
# SSNPP: Facebook SimSearchNet++
###
# CONVERGED
# python algorithms/sharding/kanndi/shard_by_distance.py --input_file /datadrive/big-ann-benchmarks/data/FB_ssnpp/FB_ssnpp_database.u8bin.crop_nb_100000000 --output_dir /datadrive/big-ann/index/ssnpp/data.100M/ -M 100 --dtype uint8
