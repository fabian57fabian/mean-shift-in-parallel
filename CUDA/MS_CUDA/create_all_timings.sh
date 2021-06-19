#!/bin/bash
echo "POINTS_NUMBER,TILE_WIDTH,NAIVE,SHARED"
#dirs=( $( ls ../../datas ) )
dirs=( $( ls /MS_CUDA/datas ) )
for dir in "${dirs[@]}"
do
for thr in 8 16 32 64 128 256 512 1024
do
	echo "#ifndef THREAD_SETTINGS
#define THREAD_SETTINGS
namespace threads_settings 
{
const int THREADS = "$thr";
const int POINTS_NUMBER = "$dir";
}
#endif" > thread_settings.h
    ./build.sh &> /dev/null
    printf $dir","
    printf $thr","
    ./execute_mean_shift.sh info
done
done