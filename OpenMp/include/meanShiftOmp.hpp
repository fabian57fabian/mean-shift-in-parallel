#ifndef MS_MEANSHIFT_HPP
#define MS_MEANSHIFT_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>
#include "utils.hpp"
#include "geometricFunction.hpp"

namespace ms
{
    namespace omp
    {
        namespace stat
        {
            template <typename T, const size_t N, const size_t D>
            std::vector<utils::vec<T, D>> cluster_points(utils::mat<T, N, D> &data, const size_t niter, \
            float bandwidth, const float radius, const float min_distance, const size_t num_threads)
            {
                const float double_sqr_bdw = 2 * bandwidth * bandwidth;
                utils::mat<T, N, D> new_data;
                for (size_t i = 0; i < niter; ++i) // number_iterations
                {
                    #pragma omp parallel for default(none) \ 
                    shared(data, niter, bandwidth, radius, double_sqr_bdw, new_data) \ 
                    schedule(static) num_threads(num_threads)
                    for (size_t p = 0; p < N; ++p)
                    {
                        utils::vec<T, D> new_position{};
                        float sum_weights = 0.;
                        for (size_t q = 0; q < N; ++q)
                        {
                            double distance = geometricFunction::calc_distance(data[p], data[q]);
                            if (distance <= radius)
                            {
                                float gaussian = std::exp(-distance / double_sqr_bdw);
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

        } // namespace ms::stat

        namespace dyn
        {
            template <typename T, const size_t N, const size_t D>
            std::vector<utils::vec<T, D>> cluster_points(utils::mat<T, N, D> &data, const size_t niter, \ 
            const float bandwidth, const float radius, const float min_distance)
            {
                const float double_sqr_bdw = 2 * bandwidth * bandwidth;
                utils::mat<T, N, D> new_data;
                for (size_t i = 0; i < niter; ++i)
                {
                    #pragma omp parallel for default(none) \ 
                    shared(data, niter, bandwidth, radius, double_sqr_bdw, new_data) \ 
                    schedule(dynamic) 
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

        } // namespace ms::dyn

    } // namespace omp

} // namespace ms

#endif //MS_MEANSHIFT_HPP