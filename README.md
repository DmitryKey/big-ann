# big-ann

Billion-Scale Approximate Nearest Neighbor Search Challenge: http://big-ann-benchmarks.com/index.html

The following blog describes BuddyPQ algorithm in detail, which achived 12% increase in recall over FAISS for 10M dataset:

[Billion-Scale Vector Search: Team Sisu and BuddyPQ](https://dmitry-kan.medium.com/billion-scale-vector-search-team-sisu-and-buddypq-ce9b016fd433)

Solutions implemented so far
==
1. [**Sharding**](src/algorithms/sharding)
   1. [**Clustering**](src/algorithms/sharding/clustering) K-Means based sharding algorithm
   2. [**KANNDI**](src/algorithms/sharding/kanndi) K Approximate Nearest Neighbours DIstance-based algorithm

Related projects
==
* Billion-Scale ANN Benchmarks: https://github.com/harsha-simhadri/big-ann-benchmarks
* Million-Scale ANN Benchmarks: https://github.com/erikbern/ann-benchmarks
* Getting practical with vector search in Solr and Elasticsearch: https://github.com/DmitryKey/bert-solr-search