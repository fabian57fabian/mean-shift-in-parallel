#!/bin/bash
dirs=( $( ls ../../datas ) )
for dir in "${dirs[@]}"
do
	echo "#ifndef THREAD_SETTINGS
#define THREAD_SETTINGS
namespace threads_settings 
{
const int THREADS = 128;
const int POINTS_NUMBER = "$dir";
}
#endif" > thread_settings.h
    echo "Running mean shift for "$dir
    ./build.sh &> /dev/null
    ./execute_mean_shift.sh info
done