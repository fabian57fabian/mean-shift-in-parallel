#include <chrono>
#include <cuda.h>
#include <iostream>
#include "utils.h"
#include <stdio.h>
#include <iomanip>

//Printing params
const int CONSOLE_WIDTH = 57;

// Mean shift params
const float RADIUS = 60;
const float SIGMA = 4;
const float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
const float MIN_DISTANCE = 60;
const size_t NUM_ITER = 50;
const float DIST_TO_REAL = 10;

// Dataset
const int D = 2;
const int CENTROIDS_NUMBER = 3;
const int POINTS_NUMBER = 10000;
// Device
const int THREADS = 1024;
const int BLOCKS = (POINTS_NUMBER + THREADS - 1) / THREADS;
const int TILE_WIDTH = THREADS;

__global__ void mean_shift_tiling(const float* data, float* data_next) {

    // Shared memory allocation
    __shared__ float local_data[TILE_WIDTH * D];
    __shared__ float valid_data[TILE_WIDTH];
    // A few convenient variables
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = tid * D;
    int local_row = threadIdx.x * D;
    float new_position[D] = {0.};
    float tot_weight = 0.;
    // Load data in shared memory
    for (int t = 0; t < BLOCKS; ++t) {
        int tid_in_tile = t * TILE_WIDTH + threadIdx.x;
        if (tid_in_tile < POINTS_NUMBER) {
            int row_in_tile = tid_in_tile * D;
            for (int j = 0; j < D; ++j) {
                local_data[local_row + j] = data[row_in_tile + j];
            }
            valid_data[threadIdx.x] = 1;
        }
        else {
            for (int j = 0; j < D; ++j) {
                local_data[local_row + j] = 0;
                valid_data[threadIdx.x] = 0;
            }
        }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; ++i) {
            int local_row_tile = i * D;
            float valid_radius = RADIUS * valid_data[i];
            float sq_dist = 0.;
            for (int j = 0; j < D; ++j) {
                sq_dist += (data[row + j] - local_data[local_row_tile + j]) * (data[row + j] - local_data[local_row_tile + j]);
            }
            if (sq_dist <= valid_radius) {
                float weight = expf(-sq_dist / DBL_SIGMA_SQ);
                for (int j = 0; j < D; ++j) {
                    new_position[j] += (weight * local_data[local_row_tile + j]);
                }
                tot_weight += (weight * valid_data[i]);
            }
        }
        __syncthreads();
    }
    if (tid < POINTS_NUMBER) {
        for (int j = 0; j < D; ++j) {
            data_next[row + j] = new_position[j] / tot_weight;
        }
    }
    return;
}

std::string separation_line(){
    return std::string(CONSOLE_WIDTH, '-');
}

std::string console_log(std::string log){
    return "|" + log + std::string(CONSOLE_WIDTH-log.length()-2, ' ') + "|";
}

std::string console_log_time(std::string log, const std::chrono::duration<double, std::milli> duration){
    return console_log(log + std::to_string(duration.count()) + "ms");
}

int main() {
    // Print useful infos
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

    float *dev_data;
    float *dev_data_next;

    // Allocate GPU memory
    size_t data_bytes = POINTS_NUMBER * D * sizeof(float);
    cudaMalloc(&dev_data, data_bytes);
    cudaMalloc(&dev_data_next, data_bytes);

    // Copy to GPU memory
    const auto start_memcp = std::chrono::system_clock::now();
    cudaMemcpy(dev_data, data.data(), data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data_next, data_next.data(), data_bytes, cudaMemcpyHostToDevice);
    const std::chrono::duration<double, std::milli> duration_memcp = std::chrono::system_clock::now() - start_memcp;
    std::cout << console_log_time("Ended memcopy in ", duration_memcp) << std::endl;

    
    // Run mean shift clustering and time the execution
    std::cout << separation_line() << std::endl;
    std::cout << console_log("Starting mean shift shared memory version") << std::endl;
    const auto before = std::chrono::system_clock::now();
    for (size_t i = 0; i < NUM_ITER; ++i) {
        mean_shift_tiling<<<BLOCKS, THREADS>>>(dev_data, dev_data_next);
        cudaDeviceSynchronize();
        utils_ns::swap(dev_data, dev_data_next);
    }
    cudaMemcpy(data.data(), dev_data, data_bytes, cudaMemcpyDeviceToHost);
    const auto centroids = utils_ns::reduce_to_centroids<POINTS_NUMBER, D>(data, MIN_DISTANCE);
    const std::chrono::duration<double, std::milli> duration_mean_shift = std::chrono::system_clock::now() - before;
    std::cout << console_log_time("Shared Memory version execution in ", duration_mean_shift) << std::endl;

    // Copy from GPU and de-allocate
    cudaFree(dev_data);
    cudaFree(dev_data_next);
    std::cout << separation_line() << std::endl;
    std::cout << console_log("Centroids found:") << std::endl;
    for (const auto& c : centroids) {
        std::string xy = std::to_string(c[0]) + ", " + std::to_string(c[1]);
        std::cout << console_log(xy) << std::endl;
    }
    std::cout << separation_line() << std::endl;

    // Check if correct number
    assert(centroids.size() == CENTROIDS_NUMBER);

    // Check if these centroids are sufficiently close to real ones
    const std::array<float, CENTROIDS_NUMBER * D> real = utils_ns::load_csv<CENTROIDS_NUMBER, D>(PATH_TO_CENTROIDS, ',');
    const bool are_close = utils_ns::are_close_to_real<CENTROIDS_NUMBER, D>(centroids, real, DIST_TO_REAL);
    assert(are_close);
    const std::chrono::duration<double, std::milli> duration_all = std::chrono::system_clock::now() - start_prog;
    std::cout << console_log_time("PROCESS ENDED in ", duration_all) << std::endl;
    std::cout << separation_line() << std::endl;
    return 0;
}