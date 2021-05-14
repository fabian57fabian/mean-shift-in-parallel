#include <chrono>
#include <cuda.h>
#include <iostream>
#include "utils.h"

// Hyperparameters
const float RADIUS = 60;
const float SIGMA = 4;
const float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
const float MIN_DISTANCE = 60;
const size_t NUM_ITER = 50;
const float DIST_TO_REAL = 10;

// Dataset
const int D = 2;
const int M = 3;
const int N = 500;

const std::string PATH_TO_DATA = "../../../datas/500/points.csv";
const std::string PATH_TO_CENTROIDS = "../../../datas/500/centroids.csv";

// const std::string PATH_TO_DATA = "../../datas/1000/points.csv";
// const std::string PATH_TO_CENTROIDS = "../../datas/1000/centroids.csv";


// Device
const int THREADS = 8;
const int BLOCKS = (N + THREADS - 1) / THREADS;
const int TILE_WIDTH = THREADS;

__global__ void mean_shift_naive(float *data, float *data_next) {
    size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N) {
        size_t row = tid * D;
        float new_position[D] = {0.};
        float tot_weight = 0.;
        for (size_t i = 0; i < N; ++i) {
            size_t row_n = i * D;
            float sq_dist = 0.;
            for (size_t j = 0; j < D; ++j) {
                sq_dist += (data[row + j] - data[row_n + j]) * (data[row + j] - data[row_n + j]);
            }
            if (sq_dist <= RADIUS) {
                float weight = expf(-sq_dist / DBL_SIGMA_SQ);
                for (size_t j = 0; j < D; ++j) {
                    new_position[j] += weight * data[row_n + j];
                }
                tot_weight += weight;
            }
        }
        for (size_t j = 0; j < D; ++j) {
            data_next[row + j] = new_position[j] / tot_weight;
        }
    }
    return;
}

int main() {


    utils_ns::print_info(PATH_TO_DATA, N, D, BLOCKS, THREADS);

    // Load data
    std::array<float, N * D> data = utils_ns::load_csv<N, D>(PATH_TO_DATA, ',');
    std::array<float, N * D> data_next {};
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
    const auto before = std::chrono::system_clock::now();
    for (size_t i = 0; i < NUM_ITER; ++i) {
        mean_shift_naive<<<BLOCKS, THREADS>>>(dev_data, dev_data_next);
        cudaDeviceSynchronize();
        utils_ns::swap(dev_data, dev_data_next);
    }
    cudaMemcpy(data.data(), dev_data, data_bytes, cudaMemcpyDeviceToHost);
    const auto centroids = utils_ns::reduce_to_centroids<N, D>(data, MIN_DISTANCE);
    const auto after = std::chrono::system_clock::now();
    const std::chrono::duration<double, std::milli> duration = after - before;
    std::cout << "\nNaive took " << duration.count() << " ms\n" << std::endl;

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