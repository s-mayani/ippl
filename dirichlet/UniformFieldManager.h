#ifndef IPPL_LANDAU_DAMPING_MANAGER_H
#define IPPL_LANDAU_DAMPING_MANAGER_H

#include <memory>
#include <filesystem>

#include "AlpineManager.h"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class UniformFieldManager : public AlpineManager<T, Dim> {
private:
    Vector_t<T, Dim> Eext_m;
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;
    using BConds_t            = ippl::BConds<Field<T, Dim>, Dim>;

    UniformFieldManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                         std::string& solver_, std::string& stepMethod_)
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_) {}

    UniformFieldManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                         std::string& solver_, std::string& stepMethod_,
                         std::vector<std::string> preconditioner_params_)
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_,
                                preconditioner_params_) {}

    ~UniformFieldManager() {}

    void pre_run() override {
        Inform m("Pre Run");

        if ((this->solver_m != "FEM") && (this->solver_m != "FEM_PRECON") 
            && (this->solver_m != "CG") && (this->solver_m != "PCG")) {
            throw IpplException("UniformField",
                                "Solver incompatible with this simulation!");
        }

        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);
        this->rmin_m  = 0.0;
        this->rmax_m  = 1.0;
        this->Eext_m = {0.0, 0.0, -1};

        bool isFEM = ((this->getSolver() == "FEM") || (this->getSolver() == "FEM_PRECON"));
        
        Vector<int, Dim> nElements = this->nr_m - 1;
        if (isFEM) {
            this->hr_m = this->rmax_m / nElements;
        } else {
            this->hr_m = this->rmax_m / this->nr_m;
        }

        // Q = -\int\int f dx dv
        this->Q_m = -1.0;
        this->origin_m = this->rmin_m;
        this->dt_m   = std::min(.05, 0.5 * *std::min_element(this->hr_m.begin(), this->hr_m.end()));
        this->it_m   = 0;
        this->time_m = 0.0;

        m << "Discretization:" << endl
          << "nt " << this->nt_m << " Np= " << this->totalP_m << " grid = " << this->nr_m << endl;

        this->isAllPeriodic_m = false;

        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m));

        this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL(),
            isFEM));

        this->fcontainer_m->initializeFields(this->solver_m);

        if ((this->getSolver() == "PCG") || (this->getSolver() == "FEM_PRECON")) {
            this->setFieldSolver(std::make_shared<FieldSolver_t>(
                this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getE(),
                &this->fcontainer_m->getPhi(), this->preconditioner_params_m));
        } else {
            this->setFieldSolver(std::make_shared<FieldSolver_t>(
                this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getE(),
                &this->fcontainer_m->getPhi()));
        }

        this->fsolver_m->initSolver();

        this->setLoadBalancer(std::make_shared<LoadBalancer_t>(
            this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m));

        initializeParticles();

        static IpplTimings::TimerRef DummySolveTimer = IpplTimings::getTimer("solveWarmup");
        IpplTimings::startTimer(DummySolveTimer);

        this->fcontainer_m->getRho() = 0.0;

        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(DummySolveTimer);

        this->par2grid();

        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        IpplTimings::startTimer(SolveTimer);

        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(SolveTimer);

        this->grid2par();

        this->dump();

        m << "Done";
    }

    void initializeParticles() {
        Inform m("Initialize Particles");

        auto* mesh = &this->fcontainer_m->getMesh();
        auto* FL   = &this->fcontainer_m->getFL();

        bool isFEM = ((this->getSolver() == "FEM") || (this->getSolver() == "FEM_PRECON"));

        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);

        // Sample particle positions:
        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>(*FL, *mesh, isFEM);

        // unsigned int
        size_type totalP = this->totalP_m;

        size_type nlocal = (this->totalP_m / ippl::Comm->size());

        this->pcontainer_m->create(nlocal);

        view_type R = (this->pcontainer_m->R.getView());
        view_type P = (this->pcontainer_m->P.getView());

        Vector_t<double, Dim> pos;
        Vector_t<double, Dim> vel(0.0);

        for (unsigned d = 0; d < Dim; ++d) {
            pos[d] = 0.5;
        }

        R(0) = pos;
        P(0) = vel;

        Kokkos::fence();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(particleCreation);

        this->pcontainer_m->q = this->Q_m / totalP;

        // For FEM need an update due to node-centering, as periodic BCs mean
        // that a particle at R=0 is equivalent to R=1 so it could be on the 
        // wrong rank and needs to be sent over.
        if (isFEM) {
            this->pcontainer_m->update();
        }
        m << "particles created and initial conditions assigned " << endl;
    }

    void advance() override {
        if (this->stepMethod_m == "LeapFrog") {
            LeapFrogStep();
        } else {
            throw IpplException(TestName, "Step method is not set/recognized!");
        }
    }

    void LeapFrogStep() {
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        static IpplTimings::TimerRef PTimer              = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer              = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer         = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef SolveTimer          = IpplTimings::getTimer("solve");

        double dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        IpplTimings::startTimer(PTimer);
        auto Rview = pc->R.getView();
        auto Pview = pc->P.getView();
        auto Eview = pc->E.getView();
        
        //std::cout << "R and P before = " << Rview(0) << "," << Pview(0) << std::endl;

        Kokkos::parallel_for(
             "Kick1", pc->getLocalNum(), KOKKOS_LAMBDA(const size_t j) {
                 Pview(j)[0] += - 0.5 * dt * (Eview(j)[0] + Eext_m[0]);
                 Pview(j)[1] += - 0.5 * dt * (Eview(j)[1] + Eext_m[1]);
                 Pview(j)[2] += - 0.5 * dt * (Eview(j)[2] + Eext_m[2]);
             });
        IpplTimings::stopTimer(PTimer);

        //std::cout << "R and P kick 1 = " << Rview(0) << "," << Pview(0) << std::endl;

        // drift
        IpplTimings::startTimer(RTimer);
        pc->R = pc->R + dt * pc->P;
        IpplTimings::stopTimer(RTimer);

        //std::cout << "R and P drift " << Rview(0) << "," << Pview(0) << std::endl;

        // Since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        // scatter the charge onto the underlying grid
        this->par2grid();

        // Field solve
        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        this->grid2par();


        // kick
        IpplTimings::startTimer(PTimer);
        auto R2view = pc->R.getView();
        auto P2view = pc->P.getView();
        auto E2view = pc->E.getView();

        //std::cout << "Efield from solve = " << E2view(0) << std::endl;
        
        Kokkos::parallel_for(
             "Kick2", pc->getLocalNum(), KOKKOS_LAMBDA(const size_t j) {
                 P2view(j)[0] += - 0.5 * dt * (E2view(j)[0] + Eext_m[0]);
                 P2view(j)[1] += - 0.5 * dt * (E2view(j)[1] + Eext_m[1]);
                 P2view(j)[2] += - 0.5 * dt * (E2view(j)[2] + Eext_m[2]);
             });
        IpplTimings::stopTimer(PTimer);
        //std::cout << "R and P kick 2 = " << R2view(0) << "," << P2view(0) << std::endl;
    }

    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);

        if ((this->getSolver() == "FEM") || (this->getSolver() == "FEM_PRECON")) {
            // When using FEM, we only have E on particles
            // so we use the dump function which computes the 
            // energy using the particles instead of the field.
            dumpLandau();
            dumpParticleData();
        } else {
            //dumpLandau(this->fcontainer_m->getE().getView());
            dumpLandau();
            dumpParticleData();
        }

        IpplTimings::stopTimer(dumpDataTimer);
    }

    template <typename View>
    void dumpLandau(const View& Eview) {
        const int nghostE = this->fcontainer_m->getE().getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double localEx2 = 0, localExNorm = 0;
        ippl::parallel_reduce(
            "Ex stats", ippl::getRangePolicy(Eview, nghostE),
            KOKKOS_LAMBDA(const index_array_type& args, double& E2, double& ENorm) {
                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double val = ippl::apply(Eview, args)[0];
                double e2  = Kokkos::pow(val, 2);
                E2 += e2;

                double norm = Kokkos::fabs(ippl::apply(Eview, args)[0]);
                if (norm > ENorm) {
                    ENorm = norm;
                }
            },
            Kokkos::Sum<double>(localEx2), Kokkos::Max<double>(localExNorm));

        double globaltemp = 0.0;
        ippl::Comm->reduce(localEx2, globaltemp, 1, std::plus<double>());

        double fieldEnergy =
            std::reduce(this->fcontainer_m->getHr().begin(), this->fcontainer_m->getHr().end(),
                        globaltemp, std::multiplies<double>());

        double ExAmp = 0.0;
        ippl::Comm->reduce(localExNorm, ExAmp, 1, std::greater<double>());

        if (ippl::Comm->rank() == 0) {
            std::filesystem::create_directory("data");
            std::stringstream fname;
            fname << "data/FieldLandau_";
            fname << ippl::Comm->size();
            fname << "_manager";
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if (std::fabs(this->time_m) < 1e-14) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }
            csvout << this->time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }
        ippl::Comm->barrier();
    }

    // Overloaded dumpLandau which computes the E-field energy using the particles 
    // instead of using the E-field on the grid (as above). Since we have E for 
    // each particle, we treat the particles as Monte-Carlo samples to compute
    // the energy integral.
    void dumpLandau() {
        auto Eview = this->pcontainer_m->E.getView();
        size_type localParticles = this->pcontainer_m->getLocalNum();

        using exec_space = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;
        policy_type iteration_policy(0, localParticles);

        double localEx2 = 0;
        Kokkos::parallel_reduce(
            "Ex stats", iteration_policy,
            KOKKOS_LAMBDA(const size_t i, double& E2) {
                double val = Eview(i)[0];
                double e2  = Kokkos::pow(val, 2);
                E2 += e2;
            },
            Kokkos::Sum<double>(localEx2));

        double globaltemp = 0.0;
        ippl::Comm->reduce(localEx2, globaltemp, 1, std::plus<double>());

        // MC integration: divide by no. of particles N and multiply by volume
        ippl::Vector<T, Dim> domain_size = this->rmax_m - this->rmin_m;
        double fieldEnergy =
            std::reduce(domain_size.begin(), domain_size.end(),
                        globaltemp, std::multiplies<double>());

        fieldEnergy = fieldEnergy / this->totalP_m;

        if (ippl::Comm->rank() == 0) {
            std::filesystem::create_directory("data");
            std::stringstream fname;
            fname << "data/FieldLandau_";
            fname << ippl::Comm->size();
            fname << "_manager";
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if (std::fabs(this->time_m) < 1e-14) {
                csvout << "time, Ex_field_energy" << endl;
            }
            csvout << this->time_m << " " << fieldEnergy << endl;
        }
        ippl::Comm->barrier();
    }

    void dumpParticleData() {
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        typename ParticleAttrib<Vector_t<T, Dim>>::HostMirror R_host = pc->R.getHostMirror();
        typename ParticleAttrib<Vector_t<T, Dim>>::HostMirror P_host = pc->P.getHostMirror();
        Kokkos::deep_copy(R_host, pc->R.getView());
        Kokkos::deep_copy(P_host, pc->P.getView());

        std::stringstream pname;
        pname << "data/Particle_";
        pname << ippl::Comm->rank();
        pname << ".csv";
        Inform pcsvout(NULL, pname.str().c_str(), Inform::APPEND, ippl::Comm->rank());
        pcsvout.precision(10);
        pcsvout.setf(std::ios::scientific, std::ios::floatfield);

        std::cout << "time = " << this->time_m << std::endl;

        if (std::fabs(this->time_m) < 1e-14) {
            pcsvout << "time, R_x, R_y, R_z, V_x, V_y, V_z" << endl;
        }
            
        pcsvout << this->time_m << " ";
        for (unsigned d = 0; d < Dim; d++) {
            pcsvout << R_host(0)[d] << " ";
        }
        for (unsigned d = 0; d < Dim; d++) {
            pcsvout << P_host(0)[d] << " ";
        }
        pcsvout << endl;
        ippl::Comm->barrier();
    }
};
#endif
