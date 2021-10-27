# fix the util package reading issue
export PYTHONPATH=.
###
# BIGANN
###

# 100M points
#python algorithms/sharding/kanndi/shard_by_distance.py --input_file /datadrive/big-ann-benchmarks/data/bigann.bak/base.1B.u8bin.crop_nb_100000000 --output_dir /datadrive/big-ann/index/bigann/data.100M -M 100

# 1B points
python algorithms/sharding/kanndi/shard_by_distance.py --input_file /datadrive/big-ann-benchmarks/data/bigann/base.1B.u8bin --output_dir /datadrive/big-ann/index/bigann/data.1B/ -M 1000 --dtype uint8


###
# Text2Image
###
#python algorithms/sharding/kanndi/shard_by_distance.py --input_file /datadrive/big-ann-benchmarks/data/text2image1B/base.1B.fbin.crop_nb_100000000 --output_dir /datadrive/big-ann/text2image/data.1B/ -M 100

###
# SSNPP: Facebook SimSearchNet++
###
# CONVERGED
# python algorithms/sharding/kanndi/shard_by_distance.py --input_file /datadrive/big-ann-benchmarks/data/FB_ssnpp/FB_ssnpp_database.u8bin.crop_nb_100000000 --output_dir /datadrive/big-ann/index/ssnpp/data.100M/ -M 100 --dtype uint8
