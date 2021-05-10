#/bin/bash
WORKING_DIR="datas"
echo "Removing temporany datas"
if [ -d "$WORKING_DIR" ]; then rm $WORKING_DIR -r; fi
echo "Downloading datas"
wget https://github.com/fabian57fabian/mean-shift-in-parallel/releases/download/0.1/datas.tar.gz
echo "Extracting datas"
tar -zxvf datas.tar.gz -C datas/
echo "Done"