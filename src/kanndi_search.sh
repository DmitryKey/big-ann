# fix the util package reading issue
export PYTHONPATH=.

###
# SSNPP-100M: Facebook SimSearchNet++
###
python algorithms/sharding/kanndi/kanndi_search.py --query_file /datadrive/big-ann-benchmarks/data/FB_ssnpp/FB_ssnpp_public_queries.u8bin --index_dir ../index/ssnpp/data.100M/ --dtype uint8

