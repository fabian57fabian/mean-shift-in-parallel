#include <chrono>
#include <cuda.h>
#include <iostream>
#include "utils.h"
#include "thread_settings.h"
#include <stdio.h>
#include <iomanip>
#include <array>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

// Mean shift params
const float RADIUS = 60;
const float SIGMA = 4;
const float BANDWIDTH = (2 * SIGMA * SIGMA);
const float MIN_DISTANCE = 60;
const size_t NUM_ITER = 50;
const float EPSILON_CHECK_CENTROIDS = 10;

// Dataset
const int CENTROIDS_NUMBER = 3;
const int POINTS_NUMBER = threads_settings::POINTS_NUMBER;

// Device
const int THREADS = threads_settings::THREADS;
const int TILE_WIDTH = THREADS;
const int D = 2;
const int BLOCKS = (POINTS_NUMBER + THREADS - 1) / THREADS;

// Kernel for naive version of weights computation.
__global__ void compute_weights_naive_kernel(float *data, float *data_tmp, const int POINTS_NUMBER) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int x = tid * 2; //DIM=2 so *2
    int y = x + 1;
    if (tid < POINTS_NUMBER) {
        float new_position[2] = {0.}; //DIM=2 so 2
        float tot_weight = 0., eucl_dist=0.;
        int xloop, yloop;
        float weight;
        for (int i = 0; i < POINTS_NUMBER; ++i) {
            xloop = i * 2; //DIM=2 so *2
            yloop = xloop + 1;
            eucl_dist = 0.;

            eucl_dist += (data[x] - data[xloop]) * (data[x] - data[xloop]) 
            + (data[y] - data[yloop]) * (data[y] - data[yloop]);

            if (eucl_dist <= RADIUS) {
                weight = expf(-eucl_dist / BANDWIDTH);
                new_position[0] += weight * data[xloop];
                new_position[1] += weight * data[yloop];
                tot_weight += weight;
            }
        }
        data_tmp[x] = new_position[0] / tot_weight;
        data_tmp[y] = new_position[1] / tot_weight;
    }
    return;
}

// Kernel for shared memory tiling version of weights computation
__global__ void compute_weights_shared_mem_kernel(const float* data, float* data_next, const int POINTS_NUMBER, const int BLOCKS) {

    // Shared memory allocation
    __shared__ float local_data[TILE_WIDTH * 2]; // Keeps current x,y locations for shared memory DIM=2 so *2
    __shared__ float flag_data[TILE_WIDTH];      // Keeps track of used data (0 if has data, o otherwise)

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int x = tid * 2; //DIM=2 so *2
    int y = x + 1;
    int x_local = threadIdx.x * 2; //DIM=2 so *2
    int y_local = x_local + 1;
    float new_position[2] = {0.};
    float tot_weight = 0.;

    int tid_in_tile, x_in_tile, y_in_tile;
    for (int t = 0; t < BLOCKS; ++t) {
		// Load data in shared memory
        tid_in_tile = t * TILE_WIDTH + threadIdx.x;
        if (tid_in_tile < POINTS_NUMBER) {
            x_in_tile = tid_in_tile * 2; //DIM=2 so *2
			y_in_tile = x_in_tile + 1;

            local_data[x_local] = data[x_in_tile];//D=0
            local_data[y_local] = data[y_in_tile];//D=1
            flag_data[threadIdx.x] = 1;
        }
        else {
            local_data[x_local] = 0;//D=0
            local_data[y_local] = 0;//D=1
            flag_data[threadIdx.x] = 0;
        }
        __syncthreads();
		// Computes mean shift weights on shared memory datas
        int local_x_tile, local_y_tile;
        float valid_radius, eucl_dist, weight;
        for (int i = 0; i < TILE_WIDTH; ++i) {
            local_x_tile = i * 2; //DIM=2 so *2
            valid_radius = RADIUS * flag_data[i];
            eucl_dist = 0.;

            eucl_dist += (data[x] - local_data[local_x_tile]) * (data[x] - local_data[local_x_tile]) 
            + (data[y] - local_data[local_y_tile]) * (data[y] - local_data[local_y_tile]);

            if (eucl_dist <= valid_radius) {
                weight = expf(-eucl_dist / BANDWIDTH);
                new_position[0] += (weight * local_data[local_x_tile]);
                new_position[1] += (weight * local_data[local_y_tile]);
                tot_weight += (weight * flag_data[i]);
            }
        }
        __syncthreads();
    }
    // Set datas for next iteration
    if (tid < POINTS_NUMBER) {
        data_next[x] = new_position[0] / tot_weight;
        data_next[y] = new_position[1] / tot_weight;
    }
    return;
}


// Functions for console printing
#define CONSOLE_WIDTH 57
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
// End functions for console printing

template <const size_t N, const size_t D>
std::vector<std::array<float, D>> reduce_to_centroids(std::array<float, N * D>& data, const float min_distance) {
    std::vector<std::array<float, D>> centroids;
    centroids.reserve(4);
    std::array<float, D> first_centroid;
    first_centroid[0] = data[0];
    first_centroid[1] = data[1];
    centroids.emplace_back(first_centroid);
    for (size_t i = 0; i < N; ++i) {
        bool at_least_one_close = false;
        for (const auto& c : centroids) {
            float dist = 0;
            for (size_t j = 0; j < D; ++j) {
                dist += ((data[i * D + j] - c[j])*(data[i * D + j] - c[j]));
            }
            if (dist <= min_distance) {
                at_least_one_close = true;
            }
        }
        if (!at_least_one_close) {
            std::array<float, D> centroid;
            centroid[0] = data[i * D];
            centroid[1] = data[i * D + 1];
            centroids.emplace_back(centroid);
        }
    }
    return centroids;
}

template <const size_t M, const size_t D>
bool are_close_to_real(const std::vector<std::array<float, D>>& centroids,
                       const std::array<float, M * D>& real,
                       const float eps_to_real) {
    std::array<bool, M> are_close {false};
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < M; ++j) {
            float dist = 0;
            for (size_t k = 0; k < D; ++k) {
                dist += ((centroids[i][k] - real[j * D + k])*(centroids[i][k] - real[j * D + k]));
            }
            if (dist <= eps_to_real) {
                are_close[i] = true;
            }
        }
    }
    return std::all_of(are_close.begin(), are_close.end(), [](const bool b){return b;}); 
}

int execute_mean_shift(bool USE_SHARED, bool DEBUG) {

    // Print useful infos
    if (DEBUG) std::cout << separation_line() << std::endl;
    if (DEBUG) std::cout << console_log(USE_SHARED?"CUDA MEAN SHIFT: SHARED MEMORY":"CUDA MEAN SHIFT: NAIVE") << std::endl;
    if (DEBUG) std::cout << separation_line() << std::endl;
    if (DEBUG) std::cout << "|POINTS_NUMBER\t|BLOCKS\t|THREADS\t|" << (USE_SHARED?"TILE_WIDTH":"         ") << "\t|"<<std::endl;
    if (DEBUG) std::cout << "|" << POINTS_NUMBER << "      \t|" << BLOCKS << "\t|" << THREADS << "      \t|" << (USE_SHARED?std::to_string(TILE_WIDTH):"    ") << "      \t!"<<std::endl;
    if (DEBUG) std::cout << separation_line() << std::endl;

    //Compute paths
    const std::string PATH_TO_DATA = "../../datas/"+std::to_string(POINTS_NUMBER)+"/points.csv";
    const std::string PATH_TO_CENTROIDS = "../../datas/"+std::to_string(POINTS_NUMBER)+"/centroids.csv";

    const auto start_prog = std::chrono::system_clock::now();

    // Load data
    if (DEBUG) std::cout << console_log("Loading csv...") << std::endl;
    std::array<float, POINTS_NUMBER * D> data = utils_ns::load_csv<POINTS_NUMBER, D>(PATH_TO_DATA, ',');
    std::array<float, POINTS_NUMBER * D> data_next {};
    if (DEBUG) std::cout << console_log("Done") << std::endl;
    if (DEBUG) std::cout << separation_line() << std::endl;

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
    if (DEBUG) std::cout << console_log_time("Ended memcopy in ", duration_memcp) << std::endl;


    // Run mean shift clustering and time the execution
    if (DEBUG) std::cout << separation_line() << std::endl;
    if (DEBUG) std::cout << console_log("Executing mean shift...") << std::endl;

    const auto starting_mean_shift_time = std::chrono::system_clock::now();
    float *temp_data;
    for (size_t i = 0; i < NUM_ITER; ++i) {
        if(USE_SHARED){
            compute_weights_shared_mem_kernel<<<BLOCKS, THREADS>>>(dev_data, dev_data_tmp, POINTS_NUMBER, BLOCKS);
        }else{
            compute_weights_naive_kernel<<<BLOCKS, THREADS>>>(dev_data, dev_data_tmp, POINTS_NUMBER);
        }
        cudaDeviceSynchronize();
        temp_data = dev_data;
        dev_data = dev_data_tmp;
        dev_data_tmp = temp_data;
    }
    cudaMemcpy(data.data(), dev_data, data_bytes, cudaMemcpyDeviceToHost);
    const auto centroids = reduce_to_centroids<POINTS_NUMBER, D>(data, MIN_DISTANCE);
    const std::chrono::duration<double, std::milli> duration_mean_shift = std::chrono::system_clock::now() - starting_mean_shift_time;
    if (DEBUG) std::cout << console_log_time("Duration: ", duration_mean_shift) << std::endl; else std::cout << duration_mean_shift.count();

    // Copy from GPU and de-allocate
    cudaFree(dev_data);
    cudaFree(dev_data_tmp);
    if (DEBUG){
        std::cout << separation_line() << std::endl;
        std::cout << console_log("Centroids found:") << std::endl;
        for (const auto& c : centroids) {
            std::string xy = std::to_string(c[0]) + ", " + std::to_string(c[1]);
            std::cout << console_log(xy) << std::endl;
        }
        std::cout << separation_line() << std::endl;
    }

    // Check if correct number
    if (centroids.size() != CENTROIDS_NUMBER){
        std::cout << console_log("ERROR: resulting centroids number are different from originals!") << std::endl;
        return 1;
    }

    // Check if these centroids are sufficiently close to real ones
    const std::array<float, CENTROIDS_NUMBER * D> real = utils_ns::load_csv<CENTROIDS_NUMBER, D>(PATH_TO_CENTROIDS, ',');
    const bool are_close = are_close_to_real<CENTROIDS_NUMBER, D>(centroids, real, EPSILON_CHECK_CENTROIDS);
    if (!are_close_to_real<CENTROIDS_NUMBER, D>(centroids, real, EPSILON_CHECK_CENTROIDS)){
        std::cout << console_log("ERROR: resulting centroids are too different from originals!") << std::endl;
        return 2;
    }

    // Show execution time
    const std::chrono::duration<double, std::milli> duration_all = std::chrono::system_clock::now() - start_prog;
    if (DEBUG) std::cout << console_log_time("PROCESS ENDED in ", duration_all) << std::endl;
    if (DEBUG) std::cout << separation_line() << std::endl;
    return 0;
}


int main(int argc, char *argv[]){
    bool INFO = false;
    if(argc>1){
        INFO = strcmp( argv[1], "info") == 0;
    }
    const int res1 = execute_mean_shift(false, !INFO);
    std::cout << ",";
    const int res2 = execute_mean_shift(true, !INFO);
    std::cout << std::endl;
    return res1 + res2;
}