//
// TestHalos
//   Usage:
//     srun ./TestGaussian <nx> <ny> <nz> --info 5
//     nx        = No. cell-centered points in the x-direction
//     ny        = No. cell-centered points in the y-direction
//     nz        = No. cell-centered points in the z-direction
//
//     Example:
//       srun ./TestHalos 64 64 64 --info 5
//
//

#include "Ippl.h"

#include <nvtx3/nvToolsExt.h>

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z, double sigma = 0.05,
                                       double mu = 0.5) {
    double pi        = Kokkos::numbers::pi_v<double>;
    double prefactor = (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return prefactor * exp(-r2 / (2 * sigma * sigma));
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        const unsigned int Dim = 3;

        using Mesh_t      = ippl::UniformCartesian<double, 3>;
        using Centering_t = Mesh_t::DefaultCentering;
        typedef ippl::Field<double, Dim, Mesh_t, Centering_t> field;

        // start a timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        // start a timer
        static IpplTimings::TimerRef init = IpplTimings::getTimer("initialize");
        IpplTimings::startTimer(init);

        nvtxRangePush("initialize");

        // get the gridsize from the user
        ippl::Vector<int, Dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

        // domain
        ippl::NDIndex<Dim> owned;
        for (unsigned i = 0; i < Dim; i++) {
            owned[i] = ippl::Index(nr[i]);
        }

        // specifies decomposition; here all dimensions are parallel
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        // unit box
        double dx                        = 1.0 / nr[0];
        double dy                        = 1.0 / nr[1];
        double dz                        = 1.0 / nr[2];
        ippl::Vector<double, Dim> hr     = {dx, dy, dz};
        ippl::Vector<double, Dim> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // define the R (rho) field
        field testField;
        testField.initialize(mesh, layout);

        // set periodic boundary conditions
        typedef ippl::BConds<field, Dim> bc_type;
        bc_type bcField;
        for (unsigned int i = 0; i < 6; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<field>>(i);
        }
        testField.setFieldBC(bcField);

        // assign the rho field with a gaussian
        auto view    = testField.getView();
        const int nghost = testField.getNghost();
        const auto& ldom = layout.getLocalNDIndex();

        Kokkos::parallel_for(
            "Assign field", testField.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local to global indices
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                // define the physical points (cell-centered)
                double x = (ig + 0.5) * hr[0] + origin[0];
                double y = (jg + 0.5) * hr[1] + origin[1];
                double z = (kg + 0.5) * hr[2] + origin[2];

                view(i, j, k) = gaussian(x, y, z);
            });
        nvtxRangePop();
        IpplTimings::stopTimer(init);

        static IpplTimings::TimerRef accumulate = IpplTimings::getTimer("accumulateHalo");
        for (int i = 0; i < 4; i++) {
            // start a timer
            IpplTimings::startTimer(accumulate);
            nvtxRangePush("accumulateHalo");

            testField.accumulateHalo();
            
            nvtxRangePop();
            IpplTimings::stopTimer(accumulate);
        }

        // stop the timers
        IpplTimings::stopTimer(allTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
