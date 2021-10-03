IP=168.63.78.34
KEY=/Users/dmitry/Desktop/BigANN/Azure/dmitrykan_key.cer
SOURCE_ROOT=src
TAR_FILE=engine.tar.gz

# clean away the tar
rm $TAR_FILE

tar --exclude='src/__pycache__' -zcvf $TAR_FILE $SOURCE_ROOT
echo Uploading engine tar to remote $IP
scp -i $KEY engine.tar.gz dmitry@${IP}:/datadrive/big-ann
ssh -i $KEY dmitry@${IP} "./receive.sh"
echo Received engine tar on remote server
