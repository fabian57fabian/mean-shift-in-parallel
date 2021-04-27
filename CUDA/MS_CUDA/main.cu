#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__ void add() {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Hello from cuda thread %d \n", idx);
}

int main() {
    add<<<10,10>>>();
    cudaDeviceSynchronize();
    return 0;
}