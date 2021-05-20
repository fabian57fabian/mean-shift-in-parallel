WORKING_DIR="build"
if [ -d "$WORKING_DIR" ]; then rm $WORKING_DIR -r; fi
mkdir $WORKING_DIR
cd $WORKING_DIR
cmake ..
make
# cd ..