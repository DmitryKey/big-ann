tar --exclude='src/__pycache__' -zcf engine.tar.gz src
scp -i ~/aws/maxirwin_bigann_azure_key.pem engine.tar.gz ubuntu@40.85.81.130:/home/ubuntu/bigann/
ssh -i ~/aws/maxirwin_bigann_azure_key.pem ubuntu@40.85.81.130 "./receive.sh"