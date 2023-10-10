// Class Midpoint Quadrature

#ifndef IPPL_MIDPOINTQUADRATURE_H
#define IPPL_MIDPOINTQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    /**
     * @brief A midpoint quadrature rule with a fixed number of integration points.
     *
     * @tparam T the floating point number type used
     * @tparam NumIntegrationPoints the number of integration points. (Minimum 1)
     */
    template <typename T, unsigned NumIntegrationPoints>
    class MidpointQuadrature : public Quadrature<T, NumIntegrationPoints> {
    public:
        /**
         * @brief Construct a new Midpoint Quadrature object
         */
        MidpointQuadrature();

        /**
         * @brief Get the integration nodes for the quadrature
         *
         * @param a Lower bound of the interval
         * @param b Upper bound of the interval
         * @return std::vector<Vector<T, Dim>> - Returns a vector with number_of_points many nodes.
         */
        Vector<T, NumIntegrationPoints> getIntegrationNodes(const T& a = 0.0,
                                                            const T& b = 1.0) const override;

        /**
         * @brief Get the weights for the quadrature
         *
         * @return std::vector<T> - Returns a vector with number_of_points many weights.
         */
        Vector<T, NumIntegrationPoints> getWeights(const T& a = 0.0,
                                                   const T& b = 1.0) const override;

        /**
         * @brief Get the degree of exactness of the quadrature rule.
         *
         * @param a Lower bound of the interval
         * @param b Upper bound of the interval
         * @return unsigned - Degree of exactness
         */
        unsigned getDegree() const override;

    private:
        unsigned num_integration_points_m;
    };

}  // namespace ippl

#include "FEM/Quadrature/MidpointQuadrature.hpp"

#endif
