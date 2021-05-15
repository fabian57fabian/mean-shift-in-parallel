#/bin/bash
WORKING_DIR="datas"
TMP_FILE="datas.tar.gz"
echo "Removing temporany datas"
if [ -d "$WORKING_DIR" ]; then rm $WORKING_DIR -r; fi
if [ -d "$TMP_FILE" ]; then rm $TMP_FILE -r; fi
mkdir datas
echo "Downloading datas"
wget https://github.com/fabian57fabian/mean-shift-in-parallel/releases/download/0.3/datas.tar.gz
echo "Extracting datas"
tar -zxvf $TMP_FILE -C ./
rm $TMP_FILE
if [ -d "$TMP_FILE" ]; then rm $TMP_FILE -r; fi
echo "Done"