#include "meanShift.hpp"
#include "utils.hpp"
#include <cassert>
#include <chrono>

// Hyperparameters
const float bandwidth = 3;
const float radius = 30;
const float min_distance = 60;
const size_t niter = 50;
const double eps = 10;

// Datas
const size_t num_points = 1000;
const size_t dim = 2;
const std::string data_path = "../../datas/1000/random_pts_1k.csv";
const std::string centroids_path = "../../datas/1000/random_cts_1k.csv";

// Check
const size_t num_centroids = 3;
const double eps_to_real = 10;

int main(int argc, char const *argv[])
{
    utils::mat<float, num_points, dim> data = utils::IO::load_csv<float, num_points, dim>(data_path, ',');
    const utils::mat<float, num_centroids, dim> real_centroids = utils::IO::load_csv<float, num_centroids, dim>(centroids_path, ',');

    auto start = std::chrono::high_resolution_clock::now();
    const std::vector<utils::vec<float, dim>> centroids = ms::seq::cluster_points<float, num_points, dim>(data, niter, bandwidth, radius, min_distance);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Duration: " << duration << " ms" << std::endl;
    utils::IO::print_mat(centroids);

    std::cout << "There are " << centroids.size() << " centroids.\n";
    assert(centroids.size() == num_centroids);
    bool are_close = geometricFunction::are_close_to_real<float, num_centroids, dim>(centroids, real_centroids, eps_to_real);
    assert(are_close);

    return 0;
}
