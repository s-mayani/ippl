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
    KOKKOS_INLINE_FUNCTION Cartesian<T, Dim>::Cartesian(Vector<Kokkos::View<T*>, Dim>& spacings,
                                                        const vector_type& origin) {
        this->initialize(spacings, origin);
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION void Cartesian<T, Dim>::initialize(Vector<Kokkos::View<T*>, Dim>& spacings,
        const vector_type& origin) {

        int size_vol = 1;

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // set gridsizes
        // calculate volume size
        for (unsigned d = 0; d < Dim; ++d) {
            this->gridSizes_m[d] = spacings[d].extent(0);
            size_vol *= (this->gridSizes_m[d] - 1);
        }

        // set mesh spacing
        meshSpacing_m = spacings;

        // volume computation
        volume_m = Kokkos::View<T*>("volume", size_vol);
        Kokkos::parallel_for("set_volume", policy_type(0, size_vol), 
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
    KOKKOS_INLINE_FUNCTION void Cartesian<T, Dim>::setMeshSpacing(Vector<Kokkos::View<T*>, Dim>& spacings) {
        // assume that it still maintains gridSize of before
        for (unsigned d = 0; d < Dim; ++d) {
            if (this->gridSizes_m[d] != spacings[d].extent(0)) {
                std::cout << "Was not able to change mesh spacing, length does not match initial domain!" << std::endl;
                return;
            }
        }

        meshSpacing_m = spacings;
        this->updateCellVolume_m();
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION typename Kokkos::View<T*> Cartesian<T, Dim>::getMeshSpacing(unsigned dim) const {
        PAssert_LT(dim, Dim);
        return meshSpacing_m[dim];
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION const Vector<Kokkos::View<T*>, Dim>&
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
        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        T temp = 0;
        Kokkos::parallel_reduce("total_vol", policy_type(0, volume_m.extent(0)),
            KOKKOS_LAMBDA(unsigned int i, T& localVal) {
                localVal += volume_m(i);
            }, Kokkos::Sum<T>(temp));
        T ret = 0;
        Comm->allreduce(temp, ret, 1, std::plus<T>());
        return ret;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION void Cartesian<T, Dim>::updateCellVolume_m() {
        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // update cell volume
        Kokkos::parallel_for("update_volume", policy_type(0, volume_m.extent(0)), 
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
        Vector<size_t, Dim> cells_per_dim = this->gridSizes_m - 1;
        size_t remaining_number_of_cells = 1;
        
        for (unsigned int d = 0; d < Dim; ++d) {
            cell += ndi[d] * remaining_number_of_cells;
            remaining_number_of_cells *= cells_per_dim[d];
        }
        return cell;
    }

}  // namespace ippl
