//
// Class Cartesian
//   Cartesian class - represents uniform-spacing cartesian meshes.
//
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"

#include "Field/BareField.h"
#include "Field/Field.h"

namespace ippl {

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Cartesian<T, Dim>::Cartesian()
        : Mesh<T, Dim>() {}

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Cartesian<T, Dim>::Cartesian(const NDIndex<Dim>& ndi, const vector_type& origin) {
        this->initialize(ndi, origin);
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION void Cartesian<T, Dim>::initialize(const NDIndex<Dim>& ndi,
        const vector_type& origin) {

        int size_vol = 1;

        // set mesh spacing
        for (unsigned d = 0; d < Dim; ++d) {
            // set gridsizes
            this->gridSizes_m[d] = ndi[d].length();
            size_vol *= (this->gridSizes_m[d] - 1);

            meshSpacing_m[d] = Kokkos::View<T*>("meshSpacing", this->gridSizes_m[d]);
            ippl::parallel_for("set_spacing", (0, this->gridSizes_m[d]), 
                KOKKOS_LAMBDA(unsigned int i) {
                    meshSpacing_m[d](i) = ndi[d].stride();
            });
        }

        // volume computation
        volume_m = Kokkos::View<T*>("volume", size_vol);
        ippl::parallel_for("set_volume", (0, size_vol), 
            KOKKOS_LAMBDA(unsigned int i) {
                T val = 1;
                for (unsigned d = 0; d < Dim; ++d) {
                    val *= meshSpacing_m[d](i);
                }
                volume_m(i) = val;
        });

        // set origin
        this->setOrigin(origin);
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION void Cartesian<T, Dim>::setMeshSpacing(
        const NDIndex<Dim>& ndi) {
        // assume that ndi still maintains gridSize of before
        for (unsigned d = 0; d < Dim; ++d) {
            if (this->gridSizes_m[d] != ndi[d].length()) {
                std::cout << "Was not able to change mesh spacing, length does not match initial domain!" << std::endl;
                return;
            }
            ippl::parallel_for("set_spacing", (0, this->gridSizes_m[d]), 
                KOKKOS_LAMBDA(unsigned int i) {
                    meshSpacing_m[d](i) = ndi[d].stride();
            });
        }

        this->updateCellVolume_m();
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION view_type Cartesian<T, Dim>::getMeshSpacing(unsigned dim) const {
        PAssert_LT(dim, Dim);
        return meshSpacing_m[dim];
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION const Vector<view_type, Dim>&
    Cartesian<T, Dim>::getMeshSpacing() const {
        return meshSpacing_m;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION T Cartesian<T, Dim>::getCellVolume(const NDIndex<Dim>& ndi) const {
        size_t cell = ndindex_to_cell(ndi);
        return volume_m(cell);
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION T Cartesian<T, Dim>::getMeshVolume() const {
        T temp = 0;
        ippl::parallel_reduce("total_vol", (0, volume_m.extent(0)),
            KOKKOS_LAMBDA(unsigned int i, T localVal) {
                localVal += volume_m(i);
            }, Kokkos::Sum<T>(temp));
        Comm->allreduce
        return ret;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION void Cartesian<T, Dim>::updateCellVolume_m() {
        // update cell volume
        ippl::parallel_for("update_volume", (0, volume_m.extent(0)), 
            KOKKOS_LAMBDA(unsigned int i) {
                T val = 1;
                for (unsigned d = 0; d < Dim; ++d) {
                    val *= meshSpacing_m[d](i);
                }
                volume_m(i) = val;
        });
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION size_t Cartesian<T, Dim>::ndindex_to_cell(const NDIndex<Dim>& ndi) const {
        size_t cell = 0;
        Vector<size_t, Dim> cells_per_dim = this->gridSizes_m[d] - 1;
        size_t remainaing_number_of_cells = 1;
        
        for (unsigned int d = 0; d < Dim; ++d) {
            cell += ndi[d] * remaining_number_of_cells;
            remaining_number_of_cells *= cells_per_dim[d];
        }
        return cell;
    }

}  // namespace ippl
