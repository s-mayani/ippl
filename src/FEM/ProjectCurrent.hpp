#ifndef IPPL_PROJECT_CURRENT_H
#define IPPL_PROJECT_CURRENT_H

#include "Ippl.h"

namespace ippl {

namespace detail {
template <typename ChargeT, typename T, unsigned Dim>
struct ParticleSnapshot {
    ChargeT q;
    T x0[3];
    T x1[3];
};

template <typename Mesh, typename NedelecSpace, typename AtomicViewType, typename ChargeView>
struct AssembleCurrentDepositFunctor {
    using T         = typename Mesh::value_type;
    using indices_t = typename NedelecSpace::indices_t;
    using point_t   = typename NedelecSpace::point_t;

    static constexpr unsigned Dim     = Mesh::Dimension;
    static constexpr unsigned numDOFs = NedelecSpace::numElementDOFs;

    Vector<T, Dim> origin;
    Vector<T, Dim> h;
    NDIndex<Dim> ldom;
    AtomicViewType atomic_view;
    const NedelecSpace& space_m;
    T dt;

    AssembleCurrentDepositFunctor(Vector<T, Dim> origin_in, Vector<T, Dim> h_in, NDIndex<Dim> ldom_in,
                                  AtomicViewType atomic_view_in, const NedelecSpace& space_in,
                                  T dt_in)
        : origin(origin_in)
        , h(h_in)
        , ldom(ldom_in)
        , atomic_view(atomic_view_in)
        , space_m(space_in)
        , dt(dt_in) {}

    template <typename Q>
    KOKKOS_FUNCTION void deposit(const Vector<T, Dim>& x0, const Vector<T, Dim>& x1, const Q q) const {
        auto segs = GridPathSegmenter<Dim, T, DefaultCellCrossingRule>::split(x0, x1, origin, h);

        const T q_over_dt = static_cast<T>(q) / dt;

        for (unsigned i = 0; i < Dim + 1; ++i) {
            const auto& seg = segs[i];

            Vector<T, Dim> dp{};
            T len_sq = T(0);
            for (unsigned d = 0; d < Dim; ++d) {
                dp[d] = seg.p1[d] - seg.p0[d];
                len_sq += dp[d] * dp[d];
            }
            if (len_sq == T(0)) {
                continue;
            }

            Vector<T, Dim> mid{};
            for (unsigned d = 0; d < Dim; ++d) {
                mid[d] = T(0.5) * (seg.p0[d] + seg.p1[d]);
            }

            indices_t cellIdx{};
            for (unsigned d = 0; d < Dim; ++d) {
                cellIdx[d] = static_cast<size_t>((mid[d] - origin[d]) / h[d]);
            }

            if (!space_m.ownsElement(cellIdx)) {
                continue;
            }

            point_t xi{};
            for (unsigned d = 0; d < Dim; ++d) {
                xi[d] = (mid[d] - origin[d]) / h[d] - T(cellIdx[d]);
            }

            auto dofIdx = space_m.getFEMVectorDOFIndices(cellIdx, ldom);

            for (unsigned k = 0; k < numDOFs; ++k) {
                auto phi_k = space_m.evaluateRefElementShapeFunction(k, xi);
                T contrib  = T(0);
                for (unsigned d = 0; d < Dim; ++d) {
                    contrib += q_over_dt * dp[d] * phi_k[d];
                }
                atomic_view(dofIdx[k]) += contrib;
            }
        }
    }

    KOKKOS_FUNCTION void operator()(const int p, Kokkos::View<Vector<T, Dim>*> x0,
                                    Kokkos::View<Vector<T, Dim>*> x1, ChargeView q) const {
        deposit(x0(p), x1(p), q(p));
    }
};
}  // namespace detail

/**
 * @brief Assemble the current density RHS vector for a Nedelec FEM space.
 *
 * For each particle p moving from X0(p) to X1(p) during one time step dt,
 * the particle trajectory is split into sub-segments that each lie within a
 * single mesh cell (via GridPathSegmenter).
 * Each sub-segment's contribution to the current density is computed and then
 * scattered onto the edge DOFs of the cell that contains the sub-segment's midpoint,
 * using the Nedelec basis functions evaluated at the midpoint (equivalent to linear interpolation).
 *
 */
template <typename Mesh,
          typename ChargeAttrib,
          typename PosAttrib,
          typename FEMVector,
          typename NedelecSpace,
          typename policy_type = Kokkos::RangePolicy<>>
inline void assemble_current_nedelec(const Mesh& mesh,
                                      const ChargeAttrib& q_attrib,
                                      const PosAttrib& X0,
                                      const PosAttrib& X1,
                                      FEMVector& fem_vector,
                                      const NedelecSpace& space,
                                      policy_type iteration_policy,
                                      typename Mesh::value_type dt) {
    using T                = typename Mesh::value_type;
    constexpr unsigned Dim = Mesh::Dimension;
    using AtomicViewType   = Kokkos::View<T*, Kokkos::MemoryTraits<Kokkos::Atomic>>;
    using charge_view_type =
        std::remove_const_t<std::remove_reference_t<decltype(q_attrib.getView())>>;
    using charge_type      = typename charge_view_type::value_type;
    using ParticleSnapshot = detail::ParticleSnapshot<charge_type, T, Dim>;

    const auto origin = mesh.getOrigin();
    const auto h      = mesh.getMeshSpacing();
    auto ldom         = space.getLocalNDIndex();

    AtomicViewType atomic_view = fem_vector.getView();

    const int local_begin = static_cast<int>(iteration_policy.begin());
    const int local_end   = static_cast<int>(iteration_policy.end());
    const int local_np    = local_end - local_begin;

    detail::AssembleCurrentDepositFunctor<Mesh, NedelecSpace, AtomicViewType, charge_view_type>
        functor(origin, h, ldom, atomic_view, space, dt);

    if (!ippl::Comm || ippl::Comm->size() <= 1) {
        Kokkos::parallel_for("assemble_current_nedelec", iteration_policy,
                             KOKKOS_LAMBDA(const std::size_t p) {
                                 functor.deposit(X0(p), X1(p), q_attrib(p));
                             });
        return;
    }

    const MPI_Comm mpi_comm = static_cast<const MPI_Comm&>(*ippl::Comm);
    const int nRanks        = ippl::Comm->size();

    std::vector<ParticleSnapshot> local_snapshots(local_np);
    for (int p = 0; p < local_np; ++p) {
        const std::size_t idx  = static_cast<std::size_t>(local_begin + p);
        local_snapshots[p].q   = q_attrib(idx);
        for (unsigned d = 0; d < Dim; ++d) {
            local_snapshots[p].x0[d] = X0(idx)[d];
            local_snapshots[p].x1[d] = X1(idx)[d];
        }
    }

    std::vector<int> counts(nRanks);
    MPI_Allgather(&local_np, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi_comm);

    std::vector<int> displs(nRanks + 1, 0);
    for (int r = 0; r < nRanks; ++r) {
        displs[r + 1] = displs[r] + counts[r];
    }
    const int global_np = displs[nRanks];

    std::vector<ParticleSnapshot> global_snapshots(global_np);
    std::vector<int> byte_counts(nRanks);
    std::vector<int> byte_displs(nRanks + 1);
    const int snapshot_bytes = static_cast<int>(sizeof(ParticleSnapshot));
    for (int r = 0; r < nRanks; ++r) {
        byte_counts[r] = counts[r] * snapshot_bytes;
        byte_displs[r] = displs[r] * snapshot_bytes;
    }
    MPI_Allgatherv(local_snapshots.data(), local_np * snapshot_bytes, MPI_BYTE,
                   global_snapshots.data(), byte_counts.data(), byte_displs.data(), MPI_BYTE,
                   mpi_comm);

    Kokkos::View<Vector<T, Dim>*> x0_dev("x0", global_np);
    Kokkos::View<Vector<T, Dim>*> x1_dev("x1", global_np);
    charge_view_type q_dev("q", global_np);
    auto x0_host = Kokkos::create_mirror_view(x0_dev);
    auto x1_host = Kokkos::create_mirror_view(x1_dev);
    auto q_host  = Kokkos::create_mirror_view(q_dev);

    for (int p = 0; p < global_np; ++p) {
        q_host(p) = global_snapshots[p].q;
        for (unsigned d = 0; d < Dim; ++d) {
            x0_host(p)[d] = global_snapshots[p].x0[d];
            x1_host(p)[d] = global_snapshots[p].x1[d];
        }
    }
    Kokkos::deep_copy(x0_dev, x0_host);
    Kokkos::deep_copy(x1_dev, x1_host);
    Kokkos::deep_copy(q_dev, q_host);

    Kokkos::parallel_for(
        "assemble_current_nedelec_mpi", Kokkos::RangePolicy<>(0, global_np),
        KOKKOS_LAMBDA(const int p) { functor(p, x0_dev, x1_dev, q_dev); });
}

}  // namespace ippl
#endif
