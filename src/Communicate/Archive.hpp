//
// Class Archive
//   Class to (de-)serialize in MPI communication.
//
#include <cstring>

#include "Archive.h"

namespace ippl {
    namespace detail {

        template <typename BufferType>
        Archive<BufferType>::Archive(size_type size)
            : writepos_m(0)
            , readpos_m(0)
            , buffer_m("buffer", size) {}

        template <typename BufferType>
        template <typename T, class... ViewArgs>
        void Archive<BufferType>::serialize(const Kokkos::View<T*, ViewArgs...>& view,
                                            size_type nsends) {
            using exec_space  = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            // Extract members into local POD variables to make sure no copy constructors are
            // invoked, we use KOKKOS_LAMBDA and not KOKKOS_CLASS_LAMBDA to avoid  copying the
            // archive to the device
            constexpr size_t size    = sizeof(T);
            char* dst_ptr            = (char*)buffer_m.data();
            char* src_ptr            = (char*)view.data();
            const size_type writepos = writepos_m;
            //
            Kokkos::parallel_for(
                "Archive::serialize()", policy_type(0, nsends), KOKKOS_LAMBDA(const size_type i) {
                    std::memcpy(dst_ptr + (i * size) + writepos, src_ptr + (i * size), size);
                });
            Kokkos::fence();
            writepos_m += size * nsends;
        }

        template <typename BufferType>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<BufferType>::serialize(const Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                            size_type nsends) {
            using exec_space = typename Kokkos::View<T*, ViewArgs...>::execution_space;

            // Extract members into local POD variables to make sure no copy constructors are
            // invoked, we use KOKKOS_LAMBDA and not KOKKOS_CLASS_LAMBDA to avoid  copying the
            // archive to the device
            constexpr size_t size    = sizeof(T);
            char* dst_ptr            = (char*)buffer_m.data();
            char* src_ptr            = (char*)view.data();
            const size_type writepos = writepos_m;

            // Default index type for range policies is int64,
            // so we have to explicitly specify size_type (uint64)
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            Kokkos::parallel_for(
                "Archive::serialize()",
                // The constructor for Kokkos range policies always expects int64 regardless of
                // index type provided by template parameters, so the typecast is necessary to avoid
                // compiler warnings
                mdrange_t({0, 0}, {(long int)nsends, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    std::memcpy(dst_ptr + (Dim * i + d) * size + writepos,
                                src_ptr + (Dim * i + d) * size, size);
                });
            Kokkos::fence();
            writepos_m += Dim * size * nsends;
        }

        template <typename BufferType>
        template <typename T, class... ViewArgs>
        void Archive<BufferType>::deserialize(Kokkos::View<T*, ViewArgs...>& view,
                                              size_type nrecvs) {
            using exec_space  = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            constexpr size_t size   = sizeof(T);
            char* src_ptr           = (char*)(buffer_m.data());
            char* dst_ptr           = (char*)(view.data());
            const size_type readpos = readpos_m;

            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            Kokkos::parallel_for(
                "Archive::deserialize()", policy_type(0, nrecvs), KOKKOS_LAMBDA(const size_type i) {
                    std::memcpy(dst_ptr + i * size, src_ptr + (i * size) + readpos, size);
                });
            // Wait for deserialization kernel to complete
            // (as with serialization kernels)
            Kokkos::fence();
            readpos_m += size * nrecvs;
        }

        template <typename BufferType>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<BufferType>::deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                              size_type nrecvs) {
            using exec_space = typename Kokkos::View<T*, ViewArgs...>::execution_space;

            constexpr size_t size   = sizeof(T);
            char* src_ptr           = (char*)(buffer_m.data());
            char* dst_ptr           = (char*)(view.data());
            const size_type readpos = readpos_m;
            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            Kokkos::parallel_for(
                "Archive::deserialize()", mdrange_t({0, 0}, {(long int)nrecvs, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    std::memcpy(dst_ptr + (Dim * i + d) * size,
                                src_ptr + (Dim * i + d) * size + readpos, size);
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }
    }  // namespace detail
}  // namespace ippl
