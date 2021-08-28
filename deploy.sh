tar --exclude='src/__pycache__' -zcf engine.tar.gz src
scp -i ~/aws/maxirwin_bigann_azure_key.pem engine.tar.gz ubuntu@104.45.85.121:/home/ubuntu/bigann/
ssh -i ~/aws/maxirwin_bigann_azure_key.pem ubuntu@104.45.85.121 "./receive.sh"