#ifndef MS_GEOMETRIC_FUNCTION_HPP
#define MS_GEOMETRIC_FUNCTION_HPP

#include "utils.hpp"
#include <vector>

namespace geometricFunction
{

    template <typename T, const size_t D>
    double calc_distance(const utils::vec<T, D> &p, const utils::vec<T, D> &q)
    {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i)
            sum += ((p[i] - q[i]) * (p[i] - q[i]));
        return sum;
    }

    template <typename T, const size_t M, const size_t D>
    bool are_close_to_real(const std::vector<utils::vec<T, D>> &centroids, \ 
    const utils::mat<T, M, D> &real, const double eps_to_real)
    {
        utils::vec<bool, M> are_close{false};
        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < M; ++j)
            {
                if (calc_distance(centroids[i], real[j]) <= eps_to_real)
                    are_close[i] = true;
            }
        }
        return std::all_of(are_close.begin(), are_close.end(), [](const bool b) { return b; });
    }

    template <typename T, const size_t  D>
    bool is_centroid(std::vector<utils::vec<T, D>>& curr_centroids, \ 
    const utils::vec<T, D>& point, const double eps_clust) {
        return std::none_of(curr_centroids.begin(), 
                            curr_centroids.end(), 
                            [&](auto& c) {return calc_distance(c, point) <= eps_clust;});
    }

    template <typename T, const size_t N, const size_t D>
    std::vector<utils::vec<T, D>> reduce_to_centroids(utils::mat<T, N, D>& data, \ 
    const float min_distance) {
        std::vector<utils::vec<T, D>> centroids = {data[0]};
        for (const auto& p : data) {
            if (is_centroid(centroids, p, min_distance))
                centroids.emplace_back(p);
        }
        return centroids;
    }

} // namespace geometricFunction

#endif // MS_GEOMETRIC_FUNCTION_HPP