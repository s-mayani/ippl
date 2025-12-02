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
        typedef typename Mesh<T, Dim>::vector_type vector_type;
        typedef Cell DefaultCentering;
        using index_array_type = typename RangePolicy<Dim>::index_array_type;

        KOKKOS_INLINE_FUNCTION Cartesian();

        KOKKOS_INLINE_FUNCTION Cartesian(Vector<Kokkos::View<T*>, Dim>& spacings, const vector_type& origin);

        KOKKOS_INLINE_FUNCTION ~Cartesian() = default;

        KOKKOS_INLINE_FUNCTION void initialize(Vector<Kokkos::View<T*>, Dim>& spacings, const vector_type& origin);

        KOKKOS_INLINE_FUNCTION void setMeshSpacing(Vector<Kokkos::View<T*>, Dim>& spacings);

        KOKKOS_INLINE_FUNCTION Kokkos::View<T*> getMeshSpacing(unsigned dim) const;

        KOKKOS_INLINE_FUNCTION const Vector<Kokkos::View<T*>, Dim>& getMeshSpacing() const;

        KOKKOS_INLINE_FUNCTION T getCellVolume(const index_array_type& args) const override;

        KOKKOS_INLINE_FUNCTION T getMeshVolume() const override;

        KOKKOS_INLINE_FUNCTION void updateCellVolume_m();

        // (x,y,z) coordinates of indexed vertex:
        KOKKOS_INLINE_FUNCTION vector_type
        getVertexPosition(const NDIndex<Dim>& ndi) const override {
            using exec_space  = typename Kokkos::View<T*>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            vector_type vertexPosition;
            for (unsigned int d = 0; d < Dim; d++) {
                unsigned int idx = ndi[d].first();
                T distance = 0;
                Kokkos::parallel_reduce("sum spacings", policy_type(0, idx),
                    KOKKOS_LAMBDA(unsigned int i, T& resultLocal) {
                        resultLocal += meshSpacing_m[d](i);
                }, Kokkos::Sum<T>(distance));
                vertexPosition(d) = distance + this->origin_m(d);
            }
            return vertexPosition;
        }

        // Vertex-vertex grid spacing of indexed cell:
        KOKKOS_INLINE_FUNCTION vector_type getDeltaVertex(const NDIndex<Dim>& ndi) const override {
            using exec_space  = typename Kokkos::View<T*>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            vector_type vertexVertexSpacing;
            for (unsigned int d = 0; d < Dim; d++) {
                unsigned int i0 = ndi[d].first();
                unsigned int i1 = ndi[d].last();
                T distance = 0;
                Kokkos::parallel_reduce("sum spacings", policy_type(i0, i1),
                    KOKKOS_LAMBDA(unsigned int i, T& resultLocal) {
                        resultLocal += meshSpacing_m[d](i);
                }, Kokkos::Sum<T>(distance));
                vertexVertexSpacing[d] = distance;
            }
            return vertexVertexSpacing;
        }

        KOKKOS_INLINE_FUNCTION size_t index_to_cell(const index_array_type& args) const;

    private:
        Vector<Kokkos::View<T*>, Dim> meshSpacing_m;  // delta-x, delta-y (>1D), delta-z (>2D)
        Kokkos::View<T*> volume_m;                    // Cell length(1D), area(2D), or volume (>2D)
    };

}  // namespace ippl

#include "Meshes/Cartesian.hpp"

#endif
