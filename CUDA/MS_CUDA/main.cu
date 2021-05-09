#include <chrono>
#include <cuda.h>
#include <iostream>
#include "utils.h"
#include <stdio.h>

constexpr float RADIUS = 60;
constexpr float SIGMA = 4;
constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
constexpr float MIN_DISTANCE = 60;
constexpr size_t NUM_ITER = 50;
constexpr float DIST_TO_REAL = 10;
// Dataset
const std::string PATH_TO_DATA = "../../../datas/1000/random_pts_1k.csv";
const std::string PATH_TO_CENTROIDS = "../../datas/1000/random_cts_1k.csv";
constexpr int N = 5000;
constexpr int D = 3;
constexpr int M = 3;
// Device
constexpr int THREADS = 64;
constexpr int BLOCKS = (N + THREADS - 1) / THREADS;
constexpr int TILE_WIDTH = THREADS;

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
        if (tid_in_tile < N) {
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
    if (tid < N) {
        for (int j = 0; j < D; ++j) {
            data_next[row + j] = new_position[j] / tot_weight;
        }
    }
    return;
}

int main() {
    utils_ns::print_info(PATH_TO_DATA, N, D, BLOCKS, THREADS, TILE_WIDTH);
    std::cout << "Loading csv" << std::endl;
    // Load data
    std::array<float, N * D> data = utils_ns::load_csv<N, D>(PATH_TO_DATA, ',');
    std::array<float, N * D> data_next {};
    std::cout << "Csv loaded" << std::endl;
    float *dev_data;
    float *dev_data_next;
    // Allocate GPU memory
    size_t data_bytes = N * D * sizeof(float);
    cudaMalloc(&dev_data, data_bytes);
    cudaMalloc(&dev_data_next, data_bytes);
    // Copy to GPU memory
    cudaMemcpy(dev_data, data.data(), data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data_next, data_next.data(), data_bytes, cudaMemcpyHostToDevice);
    // Run mean shift clustering and time the execution
    std::cout << "Ended memcopy. starting ms" << std::endl;
    const auto before = std::chrono::system_clock::now();
    for (size_t i = 0; i < NUM_ITER; ++i) {
        mean_shift_tiling<<<BLOCKS, THREADS>>>(dev_data, dev_data_next);
        cudaDeviceSynchronize();
        utils_ns::swap(dev_data, dev_data_next);
    }
    cudaMemcpy(data.data(), dev_data, data_bytes, cudaMemcpyDeviceToHost);
    const auto centroids = utils_ns::reduce_to_centroids<N, D>(data, MIN_DISTANCE);
    const auto after = std::chrono::system_clock::now();
    const std::chrono::duration<double, std::milli> duration = after - before;
    std::cout << "\nShared Memory took " << duration.count() << " ms\n" << std::endl;
    // Copy from GPU and de-allocate
    cudaFree(dev_data);
    cudaFree(dev_data_next);
    utils_ns::print_data<D>(centroids);
    // Check if correct number
    assert(centroids.size() == M);
    // Check if these centroids are sufficiently close to real ones
    const std::array<float, M * D> real = utils_ns::load_csv<M, D>(PATH_TO_CENTROIDS, ',');
    const bool are_close = utils_ns::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
    assert(are_close);
    std::cout << "SUCCESS!\n";

    return 0;
}