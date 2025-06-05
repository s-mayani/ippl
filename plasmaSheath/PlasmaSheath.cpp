// Plasma Sheath simulation
//   Usage:
//     srun ./PlasmaSheath
//                  <Np> <Nt> <lbthres> <data_dir> --overallocate <ovfactor> --info 10
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     data_dir = directory where dump files are written (diagnostics)
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./PlasmaSheath 10000 10 0.01 data --overallocate 2.0 --info 10

constexpr unsigned Dim = 1;
using T                = double;
const char* TestName   = "PlasmaSheath";

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Manager/datatypes.h"

#include "Utility/IpplTimings.h"

#include "Manager/PicManager.h"
#include "PlasmaSheathManager.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);
        Inform msg2all(TestName, INFORM_ALL_NODES);

        static IpplTimings::TimerRef mainTimer       = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef initializeTimer = IpplTimings::getTimer("initialize");
        IpplTimings::startTimer(mainTimer);
        IpplTimings::startTimer(initializeTimer);

        // Read input parameters, assign them to the corresponding members of manager
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = params::nx;
        }

        int arg                 = 1;
        size_type totalP        = std::atoll(argv[arg++]);
        int nt                  = std::atoi(argv[arg++]);
        double lbt              = std::atof(argv[arg++]);
        std::string solver      = "CG";
        std::string step_method = "Boris";
        std::string directory   = argv[arg++];

        msg << "nr=" << nr << ", Np=" << totalP << ", nt=" << nt << ", solver=" << solver << endl;

        std::vector<std::string> preconditioner_params;

        // Create an instance of a manger for the considered application
        if (solver == "PCG") {
            for (int i = 0; i < 5; i++) {
                preconditioner_params.push_back(argv[arg++]);
            }
        }

        PlasmaSheathManager<T, Dim> manager(totalP, nt, nr, lbt, solver, step_method, directory,
                                            preconditioner_params);

        // Perform pre-run operations, including creating mesh, particles,...
        manager.pre_run();

        IpplTimings::stopTimer(initializeTimer);

        manager.setTime(0.0);

        msg << "Starting iterations ..." << endl;

        manager.run(manager.getNt());

        msg << "End." << endl;

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
