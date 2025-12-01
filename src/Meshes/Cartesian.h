//
// Class Cartesian
//   Cartesian class - represents cartesian meshes, allows for non-uniform spacing.
//
#ifndef IPPL_CARTESIAN_H
#define IPPL_CARTESIAN_H

#include "Meshes/CartesianCentering.h"
#include "Meshes/Mesh.h"

namespace ippl {

    template <typename T, unsigned Dim>
    class Cartesian : public Mesh<T, Dim> {
    public:
        typedef typename Mesh::vector_type vector_type;
        typedef Kokkos::View<T*> view_type;
        typedef Cell DefaultCentering;

        KOKKOS_INLINE_FUNCTION Cartesian();

        KOKKOS_INLINE_FUNCTION Cartesian(const NDIndex<Dim>& ndi, const vector_type& origin);

        KOKKOS_INLINE_FUNCTION ~Cartesian() = default;

        KOKKOS_INLINE_FUNCTION void initialize(const NDIndex<Dim>& ndii, const vector_type& origin);

        KOKKOS_INLINE_FUNCTION void setMeshSpacing(const NDIndex<Dim>& ndi);

        KOKKOS_INLINE_FUNCTION view_type getMeshSpacing(unsigned dim) const;

        KOKKOS_INLINE_FUNCTION const Vector<view_type, Dim>& getMeshSpacing() const override;

        KOKKOS_INLINE_FUNCTION T getCellVolume() const override;

        KOKKOS_INLINE_FUNCTION T getMeshVolume() const override;

        KOKKOS_INLINE_FUNCTION void updateCellVolume_m();

        // (x,y,z) coordinates of indexed vertex:
        KOKKOS_INLINE_FUNCTION vector_type
        getVertexPosition(const NDIndex<Dim>& ndi) const override {
            vector_type vertexPosition;
            for (unsigned int d = 0; d < Dim; d++) {
                unsigned int idx = ndi[d].first();
                T distance = 0;
                ippl::parallel_reduce("sum spacings", (0, idx),
                    KOKKOS_LAMBDA(unsigned int i, T& resultLocal) {
                        resultLocal += meshSpacing[d](i);
                }, Kokkos::Sum(distance));
                vertexPosition(d) = distance + this->origin_m(d);
            }
            return vertexPosition;
        }

        // Vertex-vertex grid spacing of indexed cell:
        KOKKOS_INLINE_FUNCTION vector_type getDeltaVertex(const NDIndex<Dim>& ndi) const override {
            vector_type vertexVertexSpacing;
            for (unsigned int d = 0; d < Dim; d++) {
                unsigned int i0 = ndi[d].first();
                unsigned int i1 = ndi[d].last();
                T distance = 0;
                ippl::parallel_reduce("sum spacings", (i0, i1),
                    KOKKOS_LAMBDA(unsigned int i, T& resultLocal) {
                        resultLocal += meshSpacing[d](i);
                }, Kokkos::Sum(distance));
                vertexVertexSpacing[d] = distance;
            }
            return vertexVertexSpacing;
        }

        KOKKOS_INLINE_FUNCTION size_t ndindex_to_cell(const NDIndex<Dim>& ndi) const;

    private:
        Vector<view_type, Dim> meshSpacing_m;  // delta-x, delta-y (>1D), delta-z (>2D)
        view_type volume_m;                    // Cell length(1D), area(2D), or volume (>2D)
    };

}  // namespace ippl

#include "Meshes/Cartesian.hpp"

#endif
