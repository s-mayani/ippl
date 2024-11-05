//

#ifndef IPPL_ELEMENT_H
#define IPPL_ELEMENT_H

#include "Types/Vector.h"

namespace ippl {

    /**
     * @brief Base class for all elements.
     * @tparam T The type of the coordinates of the vertices of the element.
     * @tparam Dim The dimension of the element.
     * @tparam NumVertices The number of vertices of the element.
     */
    template <typename T, unsigned Dim, unsigned NumVertices>
    class Element {
    public:
        static constexpr unsigned dim         = Dim;
        static constexpr unsigned numVertices = NumVertices;

        // A point in the local or global coordinate system
        typedef Vector<T, Dim> point_t;

        // A list of all vertices
        typedef Vector<point_t, NumVertices> vertex_points_t;

        /**
         * @brief Pure virtual function to return the coordinates of the vertices of the reference
         * element.
         *
         * @return vertex_points_t (Vector<Vector<T, Dim>, NumVertices>)
         */
        // KOKKOS_FUNCTION virtual vertex_points_t getLocalVertices() const = 0;

        /**
         * @brief Transforms a point from global to local coordinates.
         *
         * @param global_vertices A vector of the vertex indices of the global element to transform
         * to in the mesh.
         * @param point A point in global coordinates with respect to the global element.
         *
         * @return point_t
         */
        // KOKKOS_FUNCTION point_t globalToLocal(const vertex_points_t&, const point_t&) const;

        /**
         * @brief Transforms a point from local to global coordinates.
         *
         * @param global_vertices A vector of the vertex indices of the global element to transform
         * to in the mesh.
         * @param point A point in local coordinates with respect to the reference element.
         *
         * @details Equivalent to transforming a local point \f$\hat{\boldsymbol{x}}\f$ on the local
         * element \f$\hat{K}\f$ to a point in the global coordinate system \f$\boldsymbol{x}\f$ on
         * \f$K\f$ by applying the transformation \f$\mathbf{\Phi}_K\f$ \f\[\boldsymbol{x} =
         * \mathbf{\Phi}_K(\hat{\boldsymbol{x}})\f\]
         *
         * @return point_t
         */
        // KOKKOS_FUNCTION point_t localToGlobal(const vertex_points_t& global_vertices,
        //                       const point_t& point) const;

        /**
         * @brief Returns the determinant of the transformation Jacobian.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         * transform to.
         *
         * @return T - The determinant of the transformation Jacobian
         */
        // KOKKOS_FUNCTION T getDeterminantOfTransformationJacobian(
        //     const vertex_points_t& global_vertices) const;

        /**
         * @brief Returns the inverse of the transpose of the transformation Jacobian.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         *  transform to.
         *
         * @return point_t (Vector<T, Dim>) - A vector representing the diagonal elements
         * of the inverse transpose Jacobian matrix
         */
        // KOKKOS_FUNCTION point_t getInverseTransposeTransformationJacobian(
        //     const vertex_points_t& global_vertices) const;

        /**
         * @brief Returns whether a point in local coordinates ([0, 1]^Dim) is inside the reference
         * element.
         *
         * @param point A point in local coordinates with respect to the reference element.
         * @return boolean - Returns true when the point is inside the reference element or on the
         * boundary. Returns false else
         */
        // KOKKOS_FUNCTION bool isPointInRefElement(const Vector<T, Dim>& point) const;

    protected:
        /**
         * @brief Pure virtual function to return the Jacobian of the transformation matrix.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         * transform to.
         *
         * @return point_t (Vector<T, Dim>) - A vector representing the diagonal elements
         * of the Jacobian matrix
         */
        // KOKKOS_FUNCTION virtual point_t getTransformationJacobian(
        //     const vertex_points_t& global_vertices) const = 0;

        /**
         * @brief Pure virtual function to return the inverse of the Jacobian of the transformation
         * matrix.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         * transform to.
         *
         * @return point_t (Vector<T, Dim>) - A vector representing the diagonal elements
         * of the inverse Jacobian matrix
         */
        // KOKKOS_FUNCTION virtual point_t getInverseTransformationJacobian(
        //     const vertex_points_t& global_vertices) const = 0;
    };

    /**
     * @brief Base class for all 1D elements.
     * @tparam T The type of the coordinates of the vertices of the element.
     * @tparam NumVertices The number of vertices of the element.
     */
    template <typename T, unsigned NumVertices>
    using Element1D = Element<T, 1, NumVertices>;

    /**
     * @brief Base class for all 2D elements.
     * @tparam T The type of the coordinates of the vertices of the element.
     * @tparam NumVertices The number of vertices of the element.
     */
    template <typename T, unsigned NumVertices>
    using Element2D = Element<T, 2, NumVertices>;

    /**
     * @brief Base class for all 3D elements.
     * @tparam T The type of the coordinates of the vertices of the element.
     * @tparam NumVertices The number of vertices of the element.
     */
    template <typename T, unsigned NumVertices>
    using Element3D = Element<T, 3, NumVertices>;

}  // namespace ippl

#include "FEM/Elements/Element.hpp"

#endif
