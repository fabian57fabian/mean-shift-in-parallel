#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__ void add() {
    printf("CACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaaAAAAAAAAAAA");
}

int main() {
    add<<<10,10>>>();
    cudaDeviceSynchronize();
    return 0;
}