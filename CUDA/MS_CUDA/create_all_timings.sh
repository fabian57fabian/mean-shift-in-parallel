#!/bin/bash
echo "THREADS,POINTS_NUMBER,NAIVE,SHARED"
#dirs=( $( ls ../../datas ) )
dirs=( $( ls /MS_CUDA/datas ) )
for thr in 4 8 16 32 64 128 256 512 1024
do
for dir in "${dirs[@]}"
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
    printf $thr","
    printf $dir","
    ./execute_mean_shift.sh info
done
done