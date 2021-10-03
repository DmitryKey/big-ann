tar --exclude='src/__pycache__' -zcf engine.tar.gz src/algorithms/sharding/clustering
scp -i ~/aws/maxirwin_bigann_azure_key.pem engine.tar.gz ubuntu@104.41.221.14:/home/ubuntu/bigann/
ssh -i ~/aws/maxirwin_bigann_azure_key.pem ubuntu@104.41.221.14 "./receive.sh"