#ifndef UTILS_H
#define UTILS_H

#include <array>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace utils_ns {

    template <const size_t POINTS_NUMBER, const size_t D>
    std::array<float, POINTS_NUMBER * D> load_csv(const std::string& path, const char delim) {
        std::ifstream file(path);
        std::string line;
        std::array<float, POINTS_NUMBER * D> data_matrix;
        for (size_t i = 0; i < POINTS_NUMBER; ++i) {
            std::getline(file, line);
            std::stringstream line_stream(line);
            std::string cell;
            for (size_t j = 0; j < D; ++j) {
                std::getline(line_stream, cell, delim);
                data_matrix[i * D + j] = std::stof(cell);
            }
        }
        file.close();
        return data_matrix;
    }

    template <const size_t POINTS_NUMBER, const size_t D>
    void write_csv(const std::array<float, POINTS_NUMBER * D>& data, const std::string& path, const char delim) {
        std::ofstream output(path);
        for (size_t i = 0; i < POINTS_NUMBER; ++i) {
            for (size_t j = 0; j < D - 1; ++j)
                output << data[i * D + j] << delim;
            output << data[i * D + D - 1] << '\n';
        }
        output.close();
        return;
    }

    template <typename T, const size_t K>
    void write_csv(const std::array<T, K>& data, const std::string& path, const char delim) {
        std::ofstream output(path);
        for (size_t i = 0; i < K; ++i) {
            output << data[i] << '\n';
        }
        output.close();
        return;
    }

    void swap(float* &a, float* &b){
        float *temp = a;
        a = b;
        b = temp;
        return;
    }
}

#endif