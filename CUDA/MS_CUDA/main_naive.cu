#include <chrono>
#include <cuda.h>
#include <iostream>
#include "utils.h"
#include <iomanip>

//Printing params
const int CONSOLE_WIDTH = 57;

// Mean shift params
const float RADIUS = 60;
const float SIGMA = 4;
const float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
const float MIN_DISTANCE = 60;
const size_t NUM_ITER = 50;
const float EPSILON_CHECK_CENTROIDS = 10;

// Dataset
const int D = 2;
const int CENTROIDS_NUMBER = 3;
const int POINTS_NUMBER = 10000;
// Device
const int THREADS = 8;
const int BLOCKS = (POINTS_NUMBER + THREADS - 1) / THREADS;
const int TILE_WIDTH = THREADS;

__global__ void mean_shift_naive(float *data, float *data_next) {
    size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < POINTS_NUMBER) {
        size_t row = tid * D;
        float new_position[D] = {0.};
        float tot_weight = 0.;
        for (size_t i = 0; i < POINTS_NUMBER; ++i) {
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
    std::cout << console_log("Starting mean shift naive version") << std::endl;
    const auto before = std::chrono::system_clock::now();
    bool orig_inverted = false;
    for (size_t i = 0; i < NUM_ITER; ++i) {
        if (!orig_inverted)
            mean_shift_naive<<<BLOCKS, THREADS>>>(dev_data, dev_data_tmp);
        else
            mean_shift_naive<<<BLOCKS, THREADS>>>(dev_data_tmp, dev_data);
        cudaDeviceSynchronize();
        orig_inverted = !orig_inverted;
    }
    cudaMemcpy(data.data(), orig_inverted?dev_data_tmp:dev_data, data_bytes, cudaMemcpyDeviceToHost);
    const auto centroids = utils_ns::reduce_to_centroids<POINTS_NUMBER, D>(data, MIN_DISTANCE);
    const auto after = std::chrono::system_clock::now();
    const std::chrono::duration<double, std::milli> duration_mean_shift = std::chrono::system_clock::now() - before;
    std::cout << console_log_time("Naive version execution in ", duration_mean_shift) << std::endl;

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
    std::cout << console_log_time("Process ended in in ", duration_all) << std::endl;
    std::cout << separation_line() << std::endl;
    return 0;
}