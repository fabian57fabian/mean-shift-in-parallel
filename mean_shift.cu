__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int execute_ms() {
    cuda_hello<<<1,1>>>(); 
    return 0;
}
