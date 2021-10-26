IP=23.102.16.204
KEY=/Users/dmitry/Desktop/BigANN/Azure/dmitrykan_key.cer
SOURCE_ROOT=src
TAR_FILE=engine.tar.gz
# 64G vm
# USER=dmitry
# 128G vm
USER=azureuser

# clean away the tar
rm $TAR_FILE

tar --exclude='src/__pycache__' -zcvf $TAR_FILE $SOURCE_ROOT
echo Uploading engine tar to remote $IP
scp -i $KEY engine.tar.gz ${USER}@${IP}:/datadrive/big-ann
ssh -i $KEY ${USER}@${IP} "./receive.sh"
echo Received engine tar on remote server
