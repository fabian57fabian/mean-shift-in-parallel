#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "utility.h"

__global__ void add() {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Hello from cuda thread %d \n", idx);
}

int main() {
    const std::string datas_path = "../datas/random_pts_1k.csv";
    const int N = 1000;
    const int D = 2;
    std::array<float, N * D> data = ms_utils::load_csv<N, D>(datas_path, ',');
    add<<<10,10>>>();
    cudaDeviceSynchronize();
    return 0;
}