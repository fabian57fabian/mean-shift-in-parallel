#ifndef MS_UTILS_HPP
#define MS_UTILS_HPP

#include <cstddef>
#include <array>
#include <cassert>
// #include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

namespace utils
{
    template <typename T, const size_t D>
    using vec = std::array<T, D>;

    template <typename T, const size_t N, const size_t D>
    using mat = std::array<vec<T, D>, N>;

    namespace op
    {
        template <typename T, const size_t D>
        vec<T, D> operator+(const vec<T, D> &a, const vec<T, D> &b)
        {
            vec<T, D> res;
            for (size_t i = 0; i < D; ++i)
                res[i] = a[i] + b[i];
            return res;
        }

        template <typename T, const size_t D>
        vec<T, D> operator-(const vec<T, D> &a, const vec<T, D> &b)
        {
            vec<T, D> res;
            for (size_t i = 0; i < D; ++i)
                res[i] = a[i] - b[i];
            return res;
        }

        template <typename T, const size_t D>
        vec<T, D> operator/(const vec<T, D> &a, const vec<T, D> &b)
        {
            vec<T, D> res;
            for (size_t i = 0; i < D; ++i)
                res[i] = a[i] / b[i];
            return res;
        }

        template <typename T, typename U, const size_t D>
        vec<T, D> operator/(const vec<T, D> &a, const U b)
        {
            vec<T, D> res;
            for (size_t i = 0; i < D; ++i)
                res[i] = a[i] / b;
            return res;
        }

        template <typename T, typename U, const size_t D>
        vec<T, D> operator*(const vec<T, D> &a, const U b)
        {
            vec<T, D> res;
            for (size_t i = 0; i < D; ++i)
                res[i] = a[i] * b;
            return res;
        }

    } // namespace op

    namespace IO
    {
        template <typename T, const size_t N, const size_t D>
        mat<T, N, D> load_csv(const std::string &path, const char delim)
        {
            assert(fs::exists(path));
            std::ifstream file(path);
            std::string line;
            mat<T, N, D> data_matrix;
            for (size_t i = 0; i < N; ++i)
            {
                std::getline(file, line);
                std::stringstream line_stream(line);
                std::string cell;
                vec<T, D> point;
                for (size_t j = 0; j < D; ++j)
                {
                    std::getline(line_stream, cell, delim);
                    point[j] = static_cast<T>(std::stod(cell));
                }
                data_matrix[i] = point;
            }
            file.close();
            return data_matrix;
        }

        template <typename T, const size_t N, const size_t D>
        void write_csv(const mat<T, N, D> &data, const std::string &path, const char delim)
        {
            std::ofstream output(path);
            for (size_t i = 0; i < N; ++i)
            {
                for (size_t j = 0; j < D - 1; ++j)
                    output << data[i][j] << delim;
                output << data[i][D - 1] << '\n';
            }
            output.close();
            return;
        }

        template <typename T, const size_t D>
        void print_vec(const vec<T, D> &vector)
        {
            for (auto v : vector)
                std::cout << v << ' ';
            std::cout << '\n';
            return;
        }

        template <typename T, const size_t N, const size_t D>
        void print_mat(const mat<T, N, D> &matrix)
        {
            for (const auto &vector : matrix)
                print_vec<T, D>(vector);
            std::cout << '\n';
            return;
        }

        template <typename T, const size_t D>
        void print_mat(const std::vector<vec<T, D>> &matrix)
        {
            for (const vec<T, D> &vector : matrix)
                print_vec<T, D>(vector);
            std::cout << '\n';
            return;
        }

    } // namespace IO

} // namespace utils

#endif // MS_UTILS_HPP