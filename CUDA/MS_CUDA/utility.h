#ifndef UTILS_H
    #define UTILS_H
#endif

#include <array>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <experimental/filesystem>

namespace ms_utils 
{
    template <const size_t N, const size_t D>
    std::array<float, N * D> load_csv(const std::string& path, const char delim) {
        assert(std::filesystem::exists(path));
        std::ifstream file(path);
        std::string line;
        std::array<float, N * D> data_matrix;
        for (size_t i = 0; i < N; ++i) {
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

    template <const size_t N, const size_t D>
    void write_csv(const std::array<float, N * D>& data, const std::string& path, const char delim) {
        std::ofstream output(path);
        for (size_t i = 0; i < N; ++i) {
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
}