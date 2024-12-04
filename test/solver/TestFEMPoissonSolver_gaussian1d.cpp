// Tests the FEM Poisson solver with a Gaussian source
// and homogeneous Dirichlet boundaries
//
// Usage:
//     ./TestFEMPoissonSolver_gaussian1d --info 5

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "PoissonSolvers/FEMPoissonSolver.h"

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussian1D(ippl::Vector<T, 1> x_vec, const T& sigma = 0.05, const T& mu = 0.5) {
    const T pi = Kokkos::numbers::pi_v<T>;
    const T x  = x_vec[0];

    const T prefactor = (1.0 / Kokkos::sqrt(2.0 * pi)) * (1.0 / sigma);
    const T r2        = (x - mu) * (x - mu);

    return prefactor * Kokkos::exp(-r2 / (2.0 * sigma * sigma));
}

template <typename T>
struct AnalyticSol {
    KOKKOS_FUNCTION const T operator()(ippl::Vector<T, 1> x_vec) const {
        const T pi     = Kokkos::numbers::pi_v<T>;
        const T sqrt_2 = Kokkos::sqrt(2.0);

        const T sigma = 0.05;
        const T mu    = 0.5;

        const T x = x_vec[0];
        const T r = (x - mu);

        return (-sigma / sqrt_2)
                * ((r / (sqrt_2 * sigma)) * Kokkos::erf(r / (sqrt_2 * sigma))
                + (1.0 / Kokkos::sqrt(pi)) * Kokkos::exp(-(r * r) / (2.0 * sigma * sigma)))
                + (0.5 - mu) * x + 0.5 * mu;
    }
};

template <typename T, unsigned Dim>
void testFEMSolver(const unsigned& numNodesPerDim, const T& domain_start = 0.0,
                   const T& domain_end = 1.0) {
    // start the timer
    static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("initTest");
    IpplTimings::startTimer(initTimer);

    Inform m("");
    Inform msg2all("", INFORM_ALL_NODES);

    using Mesh_t   = ippl::UniformCartesian<T, Dim>;
    using Field_t  = ippl::Field<T, Dim, Mesh_t, Cell>;
    using BConds_t = ippl::BConds<Field_t, Dim>;

    const unsigned numCellsPerDim = numNodesPerDim - 1;
    const unsigned numGhosts      = 1;

    // Domain: [-1, 1]
    const ippl::Vector<unsigned, Dim> nodesPerDimVec(numNodesPerDim);
    ippl::NDIndex<Dim> domain(nodesPerDimVec);
    ippl::Vector<T, Dim> cellSpacing((domain_end - domain_start) / static_cast<T>(numCellsPerDim));
    ippl::Vector<T, Dim> origin(domain_start);
    Mesh_t mesh(domain, cellSpacing, origin);

    // specifies decomposition; here all dimensions are parallel
    std::array<bool, Dim> isParallel;
    isParallel.fill(true);

    ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, domain, isParallel);
    Field_t lhs(mesh, layout, numGhosts);  // left hand side (updated in the algorithm)
    Field_t rhs(mesh, layout, numGhosts);  // right hand side (set once)
    Field_t sol(mesh, layout, numGhosts);  // right hand side (set once)

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<Field_t>>(i);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);

    // set rhs
    auto view_rhs = rhs.getView();
    auto view_sol = sol.getView();
    auto ldom     = layout.getLocalNDIndex();

    AnalyticSol<T> analytic;

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for(
        "Assign RHS", rhs.getFieldRangePolicy(), KOKKOS_LAMBDA(const index_array_type& args) {
            ippl::Vector<int, Dim> iVec = args - numGhosts;
            for (unsigned d = 0; d < Dim; ++d) {
                iVec[d] += ldom[d].first();
            }
            const ippl::Vector<T, Dim> x = (iVec)*cellSpacing + origin;

            apply(view_rhs, args) = gaussian1D<T>(x);
            apply(view_sol, args) = analytic(x);
        });

    IpplTimings::stopTimer(initTimer);

    // initialize the solver
    ippl::FEMPoissonSolver<Field_t, Field_t> solver(lhs, rhs);

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-13);
    params.add("max_iterations", 2000);

    solver.mergeParameters(params);

    // solve the problem
    solver.solve();

    // start the timer
    static IpplTimings::TimerRef errorTimer = IpplTimings::getTimer("computeError");
    IpplTimings::startTimer(errorTimer);

    // Compute the error
    const T relError = solver.getL2Error(analytic);

    lhs = lhs - sol;
    const T normError = norm(lhs) / norm(sol);

    m << std::setw(10) << numNodesPerDim;
    m << std::setw(25) << std::setprecision(16) << cellSpacing[0];
    m << std::setw(25) << std::setprecision(16) << relError;
    m << std::setw(25) << std::setprecision(16) << normError;
    m << std::setw(25) << std::setprecision(16) << solver.getResidue();
    m << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
    m << endl;

    IpplTimings::stopTimer(errorTimer);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("");

        using T = double;

        // start the timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        msg << std::setw(10) << "Size";
        msg << std::setw(25) << "Spacing";
        msg << std::setw(25) << "Relative Error";
        msg << std::setw(25) << "Norm Error";
        msg << std::setw(25) << "Residue";
        msg << std::setw(15) << "Iterations";
        msg << endl;

        for (unsigned n = 1 << 2; n <= 1 << 9; n = n << 1) {
            testFEMSolver<T, 1>(n, 0.0, 1.0);
        }

        // stop the timer
        IpplTimings::stopTimer(allTimer);

        // print the timers
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}