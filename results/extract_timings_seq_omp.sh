#!/bin/bash

## Prepare csv files
echo "dataset,time" > timings_seq.csv
echo "dataset,t1,t2,t3,t4,t5,t6,t7,t8" > timings_omp_static.csv
echo "dataset,time" > timings_omp_dynamic.csv

## Sequential
cd ..
cd Sequential/build

SIZE=(500 1000 2000 5000 10000)
for s in ${SIZE[@]} 
do
    export DATAS=${s}

    echo -e "\e[31m \e[1m computing Sequential with ${DATAS} points \e[0m"

    cmake .. && make
    var=$(./ms_sequential)
    echo ${DATAS},${var} >> ../../results/timings_seq.csv
done

## OMP
cd ..
cd ..
cd OpenMp/build

T=(1 2 3 4 5 6 7 8)
for s in ${SIZE[@]} 
do
    export DATAS=${s}
    svalue=${DATAS}
    dvalue=${DATAS}

    for t in ${T[@]}
    do
        export THREADS=${t}

        echo -e "\e[31m computing OMP static version with ${DATAS} points and ${THREADS} threads \e[0m"

        cmake .. && make
        svalue=(${svalue},$(./ms_omp_static))
    done

    echo -e "\e[31m \e[1m computing OMP dynamic version with ${DATAS} points \e[0m"
    dvalue=(${dvalue},$(./ms_omp_dynamic))
    
    echo ${dvalue} >> ../../results/timings_omp_dynamic.csv
    echo ${svalue} >> ../../results/timings_omp_static.csv
done
