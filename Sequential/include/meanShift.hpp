#ifndef MS_MEANSHIFT_HPP
#define MS_MEANSHIFT_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include "utils.hpp"
#include "geometricFunction.hpp"

namespace ms
{
    namespace seq
    {
        template <typename T, const size_t N, const size_t D>
        std::vector<utils::vec<T, D>> cluster_points(utils::mat<T, N, D> &data, const size_t niter, const float bandwidth, const float radius, const float min_distance)
        {
            const float double_sqr_bdw = 2 * bandwidth * bandwidth;
            utils::mat<T, N, D> new_data;
            for (size_t i = 0; i < niter; ++i)
            {
                for (size_t p = 0; p < N; ++p)
                {
                    utils::vec<T, D> new_position{};
                    float sum_weights = 0.;
                    for (size_t q = 0; q < N; ++q)
                    {
                        double dist = geometricFunction::calc_distance(data[p], data[q]);
                        if (dist <= radius)
                        {
                            float gaussian = std::exp(-dist / double_sqr_bdw);
                            // new_position = new_position + data[q] * gaussian;
                            new_position = utils::op::operator+(new_position, utils::op::operator*(data[q], gaussian));
                            sum_weights += gaussian;
                        }
                    }
                    new_data[p] = utils::op::operator/(new_position, sum_weights);
                }
                data = new_data;
            }
            return geometricFunction::reduce_to_centroids(data, min_distance);
        }
    } // namespace seq
} // namespace ms

#endif //MS_MEANSHIFT_HPP