// vim: ts=4 sw=4 et
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

#include <sstream>
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
#include <sstream>
#include <fstream>


#include "Manager/datatypes.h"

#include "Utility/IpplTimings.h"

#include "Manager/PicManager.h"
#include "PlasmaSheathManager.h"

#include "lib/argparse/include/argparse/argparse.hpp"

Inform& operator<<(Inform& m, const PlasmaSheathParams& p) {
    m << "-- PHYSICAL PARAMS\n"
      << "\tZ_i=" << p.Z_i << "\n"
      << "\tZ_e=" << p.Z_e << "\n"
      << "\tn_i0=" << p.n_i0 << "\n"
      << "\tn_e0=" << p.n_e0 << "\n"
      << "\tm_i=" << p.m_i << "\n"
      << "\tm_e=" << p.m_e << "\n"
      << "\ttau=" << p.tau << "\n"
      << "\tnu=" << p.nu << "\n"
      << "\tD_D=" << p.D_D << "\n"
      << "\tD_C=" << p.D_C << "\n"
      << "\talpha=" << p.alpha << "\n"
      << "\tphi0=" << p.phi0 << "\n"
      << "\tkinetic_electrons=" << p.kinetic_electrons << "\n"
      << "\tv_th_i=" << p.v_th_i << "\n"
      << "\tv_th_e=" << p.v_th_e << "\n"
      << "\trho_th_i=" << p.rho_th_i << "\n"
      << "\trho_th_e=" << p.rho_th_e << "\n"
      << "\tOmega_ci=" << p.Omega_ci << "\n"
      << "\tOmega_ce=" << p.Omega_ce << "\n"
      << "-- GRID PARAMS\n"
      << "\tL=" << p.L << "\n"
      << "\tf_x=" << p.f_x << "\n"
      << "\tf_t=" << p.f_t << "\n"
      << "\tCFL_max=" << p.CFL_max << "\n"
      << "\tf_ion_speedup=" << p.f_ion_speedup << "\n"
      << "\tf_v_th_safety=" << p.f_v_th_safety << "\n"
      << "\tv_max=" << p.v_max << "\n"
      << "\tv_trunc_i=" << p.v_trunc_i << "\n"
      << "\tv_trunc_e=" << p.v_trunc_e << "\n"
      << "\tdx0=" << p.dx0 << "\n"
      << "\tnx=" << p.nx << "\n"
      << "\tdx=" << p.dx << "\n"
      << "\tdt=" << p.dt << "\n"
      << "-- SIMULATION PARAMS\n"
      << "\tnum_particles=" << p.num_particles << "\n"
      << "\tnum_timesteps=" << p.num_timesteps << "\n"
      << "\tlbt=" << p.lbt << "\n"
      << "-- SOLVER PARAMS\n"
      << "\tsolver=" << p.solver << "\n"
      << "\tstep_method=" << p.step_method << "\n"
      << "-- OUTPUT PARAMS\n"
      << "\tdirectory=" << p.directory << "\n"
      << "\tdump_interval_plasma=" << p.dump_interval_plasma << "\n"
      << "\tdump_interval_particles=" << p.dump_interval_particles << "\n";

    return m;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);
        Inform msg2all(TestName, INFORM_ALL_NODES);

        static IpplTimings::TimerRef mainTimer       = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef initializeTimer = IpplTimings::getTimer("initialize");
        IpplTimings::startTimer(mainTimer);
        IpplTimings::startTimer(initializeTimer);

        int arg = 1;
        const std::string pname{"plasmaSheath"};
        argparse::ArgumentParser program(pname);

        program.add_argument("--Z_i")
            .required()
            .scan<'g', double>();
        program.add_argument("--n_i0")
            .required()
            .scan<'g', double>();
        program.add_argument("--m_e")
            .required()
            .scan<'g', double>();
        program.add_argument("--tau")
            .required()
            .scan<'g', double>();
        program.add_argument("--nu")
            .required()
            .scan<'g', double>();
        program.add_argument("--D_C")
            .required()
            .scan<'g', double>();
        program.add_argument("--alpha_deg")
            .required()
            .scan<'g', double>();
        program.add_argument("--phi0")
            .required()
            .scan<'g', double>();
        program.add_argument("--kinetic_electrons")
            .required()
            .scan<'d', int>();
        program.add_argument("--L")
            .required()
            .scan<'g', double>();
        program.add_argument("--f_x")
            .required()
            .scan<'g', double>();
        program.add_argument("--f_t")
            .required()
            .scan<'g', double>();
        program.add_argument("--CFL_max")
            .required()
            .scan<'g', double>();
        program.add_argument("--f_ion_speedup")
            .required()
            .scan<'g', double>();
        program.add_argument("--f_v_th_safety")
            .required()
            .scan<'g', double>();
        program.add_argument("--dump_interval_plasma")
            .required()
            .scan<'d', unsigned int>();
        program.add_argument("--dump_interval_particles")
            .required()
            .scan<'d', unsigned int>();
        program.add_argument("--num_particles")
            .required()
            .scan<'d', unsigned int>();
        program.add_argument("--num_timesteps")
            .required()
            .scan<'d', unsigned int>();
        program.add_argument("--lbt")
            .required()
            .scan<'g', double>();
        program.add_argument("--output")
            .required();

        std::string fname{argv[arg++]};
        std::ifstream ifile(fname);
        std::vector<std::string> param_list{{pname}};
        std::string line;
        while (std::getline(ifile, line)) {
            // ignore comment lines
            if (line.rfind(";", 0) == 0)
                continue;
            // split line by spaces
            std::istringstream iss(line);
            std::string p;
            while (std::getline(iss, p, ' '))
                param_list.push_back(p);
        }
        ifile.close();

        // msg << "listing input parameters..." << endl;
        // for (auto const& p : param_list)
        //     msg << p << endl;
        // msg << "... done listing input parameters" << endl;

        try {
            program.parse_known_args(param_list);
        }
        catch (const std::exception& err) {
            std::stringstream m;
            m << err.what() << std::endl;
            m << program;
            msg << m.str() << endl;
            std::exit(1);
        }

        const PlasmaSheathParams params(
            program.get<double>("--Z_i"), // double Z_i_,
            program.get<double>("--n_i0"), // double n_i0_,
            program.get<double>("--m_e"), // double m_e_,
            program.get<double>("--tau"), // double tau_,
            program.get<double>("--nu"), // double nu_,
            program.get<double>("--D_C"), // double D_C_,
            program.get<double>("--alpha_deg") * pi / 180.0, // double alpha_rad_,
            program.get<double>("--phi0"), // double phi0_,
            (bool)program.get<int>("--kinetic_electrons"), // bool kinetic_electrons_,
            program.get<double>("--L"), // double L_,
            program.get<double>("--f_x"), // double f_x_,
            program.get<double>("--f_t"), // double f_t_,
            program.get<double>("--CFL_max"), // double CFL_max_,
            program.get<double>("--f_ion_speedup"), // double f_ion_speedup_,
            program.get<double>("--f_v_th_safety"), // double f_v_th_safety_,
            program.get<unsigned int>("--dump_interval_plasma"), // unsigned int dump_interval_plasma_,
            program.get<unsigned int>("--dump_interval_particles"), // unsigned int dump_interval_particles_,
            program.get<unsigned int>("--num_particles"), // unsigned int num_particles_,
            program.get<unsigned int>("--num_timesteps"), // unsigned int num_timesteps_,
            program.get<double>("--lbt"), // double lbt_,
            "CG", // std::string solver_,
            "Boris", //std::string step_method_,
            program.get<std::string>("--output") // std::string directory_
        );
        msg << params << endl;

        // Read input parameters, assign them to the corresponding members of manager
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = params.nx;
        }

        // to maintain backwards compatibility (kept complaining about the constructor of AlpineManager)
        int nt = params.num_timesteps;
        size_type totalP = params.num_particles;
        double lbt = params.lbt;
        std::string solver = params.solver;
        std::string step_method = params.step_method;
        std::string directory = params.directory;
        //
        msg << "nr=" << nr << ", Np=" << totalP << ", nt=" << nt << ", solver=" << solver << endl;

        std::vector<std::string> preconditioner_params;

        // Create an instance of a manger for the considered application
        if (solver == "PCG") {
            for (int i = 0; i < 5; i++) {
                preconditioner_params.push_back(argv[arg++]);
            }
        }

        PlasmaSheathManager<T, Dim> manager(params, totalP, nt, nr, lbt, solver, step_method, directory,
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
