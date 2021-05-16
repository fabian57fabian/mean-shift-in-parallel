#include <chrono>
#include <cuda.h>
#include <iostream>
#include "utils.h"
#include <stdio.h>
#include <iomanip>

// Printing params
const int CONSOLE_WIDTH = 57;

// Mean shift params
const float RADIUS = 60;
const float SIGMA = 4;
const float SIGMA_POWER = (2 * SIGMA * SIGMA);
const float MIN_DISTANCE = 60;
const size_t NUM_ITER = 50;
const float EPSILON_CHECK_CENTROIDS = 10;

// Dataset
const int CENTROIDS_NUMBER = 3;
const int POINTS_NUMBER = 10000;

// Device
const int THREADS = 512;
const int TILE_WIDTH = THREADS;

__global__ void mean_shift_naive(float *data, float *data_tmp, const int POINTS_NUMBER) {
    size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < POINTS_NUMBER) {
        size_t row = tid * 2; //DIM=2 so *2
        float new_position[2] = {0.}; //DIM=2 so 2
        float tot_weight = 0.;
        for (size_t i = 0; i < POINTS_NUMBER; ++i) {
            size_t row_n = i * 2; //DIM=2 so *2
            float sq_dist = 0.;

            sq_dist += (data[row] - data[row_n]) * (data[row] - data[row_n]) 
            + (data[row + 1] - data[row_n + 1]) * (data[row + 1] - data[row_n + 1]);

            if (sq_dist <= RADIUS) {
                float weight = expf(-sq_dist / SIGMA_POWER);
                new_position[0] += weight * data[row_n];//D=0
                new_position[1] += weight * data[row_n + 1];//D=1
                tot_weight += weight;
            }
        }
        data_tmp[row] = new_position[0] / tot_weight;//D=0
        data_tmp[row + 1] = new_position[1] / tot_weight;//D=1
    }
    return;
}

__global__ void mean_shift_tiling(const float* data, float* data_next, const int POINTS_NUMBER, const int BLOCKS) {

    // Shared memory allocation
    __shared__ float local_data[TILE_WIDTH * 2]; //DIM=2 so *2
    __shared__ float valid_data[TILE_WIDTH];
    // A few convenient variables
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = tid * 2; //DIM=2 so *2
    int local_row = threadIdx.x * 2; //DIM=2 so *2
    float new_position[2] = {0.};
    float tot_weight = 0.;
    // Load data in shared memory
    int tid_in_tile, row_in_tile;
    for (int t = 0; t < BLOCKS; ++t) {
        tid_in_tile = t * TILE_WIDTH + threadIdx.x;
        if (tid_in_tile < POINTS_NUMBER) {
            row_in_tile = tid_in_tile * 2; //DIM=2 so *2

            local_data[local_row] = data[row_in_tile];//D=0
            local_data[local_row + 1] = data[row_in_tile + 1];//D=1
            valid_data[threadIdx.x] = 1;
        }
        else {
            local_data[local_row] = 0;//D=0
            local_data[local_row + 1] = 0;//D=1
            valid_data[threadIdx.x] = 0;
        }
        __syncthreads();
        int local_row_tile;
        float valid_radius, sq_dist, weight;
        for (int i = 0; i < TILE_WIDTH; ++i) {
            local_row_tile = i * 2; //DIM=2 so *2
            valid_radius = RADIUS * valid_data[i];
            sq_dist = 0.;

            sq_dist += (data[row] - local_data[local_row_tile]) * (data[row] - local_data[local_row_tile]) 
            + (data[row + 1] - local_data[local_row_tile + 1]) * (data[row + 1] - local_data[local_row_tile + 1]);

            if (sq_dist <= valid_radius) {
                weight = expf(-sq_dist / SIGMA_POWER);
                new_position[0] += (weight * local_data[local_row_tile]);
                new_position[1] += (weight * local_data[local_row_tile + 1]);
                tot_weight += (weight * valid_data[i]);
            }
        }
        __syncthreads();
    }
    if (tid < POINTS_NUMBER) {
        data_next[row] = new_position[0] / tot_weight;
        data_next[row + 1] = new_position[1] / tot_weight;
    }
    return;
}

std::string separation_line(){
    return std::string(CONSOLE_WIDTH, '-');
}

std::string console_log(std::string log){
    int spaces = CONSOLE_WIDTH-log.length()-2;
    return "|" + log + std::string(spaces>0?spaces:0, ' ') + "|";
}

std::string console_log_time(std::string log, const std::chrono::duration<double, std::milli> duration){
    return console_log(log + std::to_string(duration.count()) + "ms");
}

int execute_mean_shift(bool USE_SHARED) {
    const int D = 2;
    const int BLOCKS = (POINTS_NUMBER + THREADS - 1) / THREADS;

    // Print useful infos
    std::cout << separation_line() << std::endl;
    std::cout << console_log(USE_SHARED?"CUDA MEAN SHIFT: SHARED MEMORY":"CUDA MEAN SHIFT: NAIVE") << std::endl;
    std::cout << separation_line() << std::endl;
    std::cout << "|POINTS_NUMBER\t|BLOCKS\t|THREADS\t|TILE_WIDTH\t|"<<std::endl;
    std::cout << "|" << POINTS_NUMBER << "      \t|" << BLOCKS << "\t|" << THREADS << "      \t|" << TILE_WIDTH << "      \t!"<<std::endl;
    std::cout << separation_line() << std::endl;

    //Compute paths
    const std::string PATH_TO_DATA = "../../datas/"+std::to_string(POINTS_NUMBER)+"/points.csv";
    const std::string PATH_TO_CENTROIDS = "../../datas/"+std::to_string(POINTS_NUMBER)+"/centroids.csv";

    const auto start_prog = std::chrono::system_clock::now();

    // Load data
    std::cout << console_log("Loading csv...") << std::endl;
    std::array<float, POINTS_NUMBER * D> data = utils_ns::load_csv<POINTS_NUMBER, D>(PATH_TO_DATA, ',');
    std::array<float, POINTS_NUMBER * D> data_next {};
    std::cout << console_log("Done") << std::endl;
    std::cout << separation_line() << std::endl;

    // Allocate GPU memory
    float *dev_data, *dev_data_tmp;
    size_t data_bytes = POINTS_NUMBER * D * sizeof(float);
    cudaMalloc(&dev_data, data_bytes);
    cudaMalloc(&dev_data_tmp, data_bytes);

    // Copy to GPU memory
    const auto start_memcp = std::chrono::system_clock::now();
    cudaMemcpy(dev_data, data.data(), data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data_tmp, data_next.data(), data_bytes, cudaMemcpyHostToDevice);
    const std::chrono::duration<double, std::milli> duration_memcp = std::chrono::system_clock::now() - start_memcp;
    std::cout << console_log_time("Ended memcopy in ", duration_memcp) << std::endl;


    // Run mean shift clustering and time the execution
    std::cout << separation_line() << std::endl;
    std::cout << console_log("Executing mean shift...") << std::endl;

    const auto starting_mean_shift_time = std::chrono::system_clock::now();
    for (size_t i = 0; i < NUM_ITER; ++i) {
        if(USE_SHARED){
            mean_shift_tiling<<<BLOCKS, THREADS>>>(dev_data, dev_data_tmp, POINTS_NUMBER, BLOCKS);
        }else{
            mean_shift_naive<<<BLOCKS, THREADS>>>(dev_data, dev_data_tmp, POINTS_NUMBER);
        }
        cudaDeviceSynchronize();
        utils_ns::swap(dev_data, dev_data_tmp);
    }
    cudaMemcpy(data.data(), dev_data, data_bytes, cudaMemcpyDeviceToHost);
    const auto centroids = utils_ns::reduce_to_centroids<POINTS_NUMBER, D>(data, MIN_DISTANCE);
    const std::chrono::duration<double, std::milli> duration_mean_shift = std::chrono::system_clock::now() - starting_mean_shift_time;
    std::cout << console_log_time("Duration: ", duration_mean_shift) << std::endl;

    // Copy from GPU and de-allocate
    cudaFree(dev_data);
    cudaFree(dev_data_tmp);
    std::cout << separation_line() << std::endl;
    std::cout << console_log("Centroids found:") << std::endl;
    for (const auto& c : centroids) {
        std::string xy = std::to_string(c[0]) + ", " + std::to_string(c[1]);
        std::cout << console_log(xy) << std::endl;
    }
    std::cout << separation_line() << std::endl;

    // Check if correct number
    if (centroids.size() != CENTROIDS_NUMBER){
        std::cout << console_log("ERROR: resulting centroids number are different from originals!") << std::endl;
        return 1;
    }

    // Check if these centroids are sufficiently close to real ones
    const std::array<float, CENTROIDS_NUMBER * D> real = utils_ns::load_csv<CENTROIDS_NUMBER, D>(PATH_TO_CENTROIDS, ',');
    const bool are_close = utils_ns::are_close_to_real<CENTROIDS_NUMBER, D>(centroids, real, EPSILON_CHECK_CENTROIDS);
    if (!utils_ns::are_close_to_real<CENTROIDS_NUMBER, D>(centroids, real, EPSILON_CHECK_CENTROIDS)){
        std::cout << console_log("ERROR: resulting centroids are too different from originals!") << std::endl;
        return 2;
    }

    // Show execution time
    const std::chrono::duration<double, std::milli> duration_all = std::chrono::system_clock::now() - start_prog;
    std::cout << console_log_time("PROCESS ENDED in ", duration_all) << std::endl;
    std::cout << separation_line() << std::endl;
    return 0;
}


int main(int argc, char *argv[]){
    const int res1 = execute_mean_shift(false);
    std::cout << std::endl;
    std::cout << std::endl;
    const int res2 = execute_mean_shift(true);
    return res1 + res2;
}