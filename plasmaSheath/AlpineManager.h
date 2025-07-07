#ifndef IPPL_ALPINE_MANAGER_H
#define IPPL_ALPINE_MANAGER_H

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "Manager/PicManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

#include "input.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class AlpineManager : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>,
                                              FieldContainer<T, Dim>, LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;
    using Base                = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

    AlpineManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                  std::string& solver_, std::string& stepMethod_, std::string& directory_,
                  std::vector<std::string> preconditioner_params = {})
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                           LoadBalancer<T, Dim>>()
        , totalP_m(totalP_)
        , nt_m(nt_)
        , nr_m(nr_)
        , lbt_m(lbt_)
        , solver_m(solver_)
        , stepMethod_m(stepMethod_)
        , directory_m(directory_)
        , preconditioner_params_m(preconditioner_params) {}
    ~AlpineManager() {}

    size_type getTotalP() const { return totalP_m; }

    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    const std::string& getSolver() const { return solver_m; }

    void setSolver(const std::string& solver_) { solver_m = solver_; }

    double getLoadBalanceThreshold() const { return lbt_m; }

    void setLoadBalanceThreshold(double lbt_) { lbt_m = lbt_; }

    const std::string& getStepMethod() const { return stepMethod_m; }

    void setStepMethod(const std::string& stepMethod_) { stepMethod_m = stepMethod_; }

    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    double getTime() const { return time_m; }

    void setTime(double time_) { time_m = time_; }

    std::vector<std::string> getPreconditionerParams() const { return preconditioner_params_m; };

    virtual void dump() { /* default does nothing */ };

    void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }

    void post_step() override {
        // Update time
        this->time_m += this->dt_m;
        this->it_m++;

        Inform m("Post-step:");
        m << "Finished time step: " << this->it_m << " time: " << this->time_m << endl;
    }

    void grid2par() override { gatherCIC(); }

    void gatherCIC() {
        gather(this->pcontainer_m->E, this->fcontainer_m->getE(), this->pcontainer_m->R);
    }

    void par2grid() override { scatterCIC(); }

    void scatterCIC() {
        Inform m("scatter ");
        this->fcontainer_m->getRho() = 0.0;

        ippl::ParticleAttrib<double>* q          = &this->pcontainer_m->q;
        typename Base::particle_position_type* R = &this->pcontainer_m->R;
        Field_t<Dim>* rho                        = &this->fcontainer_m->getRho();
        double Q                                 = Q_m;
        Vector_t<double, Dim> hr                 = hr_m;

        scatter(*q, *rho, *R);

        // account for ghost cells only on physical domain boundaries
        // we need to do this because the cell-centered approach makes the charges
        // deposit also on ghost cells -- add this to the rho.sum()
        const auto& layout = this->fcontainer_m->getFL();
        const auto gdom    = layout.getDomain();
        const auto& ldom   = layout.getLocalNDIndex();
        const int nghost   = rho->getNghost();
        auto view          = rho->getView();

        using exec_space       = typename Field_t<Dim>::execution_space;
        using index_type       = typename ippl::RangePolicy<Dim, exec_space>::index_type;
        using index_array_type = typename ippl::RangePolicy<Dim, exec_space>::index_array_type;
        Kokkos::Array<index_type, Dim> begin, end, begin_ghost, end_ghost;

        bool addGhosts_upper = false;
        bool addGhosts_lower = false;
        for (unsigned int d = 0; d < Dim; ++d) {
            begin[d] = view.extent(d) - nghost;
            end[d]   = nghost;
            if (ldom[d].max() == gdom[d].max()) {
                end_ghost[d]    = view.extent(d);
                addGhosts_upper = true;
            }
            if (ldom[d].min() == gdom[d].min()) {
                begin_ghost[d]  = 0;
                addGhosts_lower = true;
            }
        }

        T sum_upper = 0;
        T sum_lower = 0;
        if (addGhosts_upper) {
            ippl::parallel_reduce(
                "Assign periodic field BC",
                ippl::createRangePolicy<Dim, exec_space>(begin, end_ghost),
                KOKKOS_LAMBDA(index_array_type & args, T & val) { val += ippl::apply(view, args); },
                Kokkos::Sum<T>(sum_upper));
        }
        if (addGhosts_lower) {
            ippl::parallel_reduce(
                "Assign periodic field BC",
                ippl::createRangePolicy<Dim, exec_space>(begin_ghost, end),
                KOKKOS_LAMBDA(index_array_type & args, T & val) { val += ippl::apply(view, args); },
                Kokkos::Sum<T>(sum_lower));
        }
        T globalSum_upper = 0;
        T globalSum_lower = 0;
        ippl::Comm->allreduce(sum_upper, globalSum_upper, 1, std::plus<T>());
        ippl::Comm->allreduce(sum_lower, globalSum_lower, 1, std::plus<T>());

        T rhoSum = (*rho).sum();
        rhoSum += globalSum_upper + globalSum_lower;

        // remove division by Q since quasi-neutral
        double absError = std::fabs(Q - rhoSum);

        m << absError << endl;

        size_type TotalParticles = 0;
        size_type localParticles = this->pcontainer_m->getLocalNum();

        ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

        if (ippl::Comm->rank() == 0) {
            if (TotalParticles != totalP_m || absError > 1e-10) {
                m << "Time step: " << it_m << endl;
                m << "Total particles in the sim. " << totalP_m << " "
                  << "after update: " << TotalParticles << endl;
                m << "Abs. error in charge conservation: " << absError << endl;
                ippl::Comm->abort();
            }
        }

        double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());
        (*rho)            = (*rho) / cellVolume;

        rhoNorm_m = norm(*rho);
    }

    void dumpCharges() {
        typename Base::particle_position_type* R = &this->pcontainer_m->R;
        ippl::ParticleAttrib<double>* q_e        = &this->pcontainer_m->q_e;
        ippl::ParticleAttrib<double>* q_i        = &this->pcontainer_m->q_i;
        Field_t<Dim>* rho                        = &this->fcontainer_m->getRho();
        Vector_t<double, Dim> hr                 = hr_m;

        auto& mesh = rho->get_mesh();
        auto& layout = this->fcontainer_m->getFL();

        double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());

        // scatter only electrons
        Field_t<Dim> rho_e(mesh, layout);
        scatter(*q_e, rho_e, *R);
        rho_e = rho_e / cellVolume;

        // scatter only ions
        Field_t<Dim> rho_i(mesh, layout);
        scatter(*q_i, rho_i, *R);
        rho_i = rho_i / cellVolume;

        // compute total sum to compare with actual rho
        Field_t<Dim> sumTotal(mesh, layout);
        sumTotal = rho_e + rho_i;

        // dump rho_e, rho_i, sum, and rho
        typename Field_t<Dim>::view_type::host_mirror_type host_view_rho_e =
            rho_e.getHostMirror();
        Kokkos::deep_copy(host_view_rho_e, rho_e.getView());

        typename Field_t<Dim>::view_type::host_mirror_type host_view_rho_i =
            rho_i.getHostMirror();
        Kokkos::deep_copy(host_view_rho_i, rho_i.getView());

        typename Field_t<Dim>::view_type::host_mirror_type host_view_sum =
            sumTotal.getHostMirror();
        Kokkos::deep_copy(host_view_sum, sumTotal.getView());

        typename Field_t<Dim>::view_type::host_mirror_type host_view_rho =
            rho->getHostMirror();
        Kokkos::deep_copy(host_view_rho, rho->getView());

        const double nx     = nr_m[0];
        const double orig_x = origin_m[0]; 
        const double hx     = hr_m[0];
        const int nghost = rho->getNghost();

        std::stringstream fname;
        fname << this->directory_m << "/Rho_";
        fname << this->it_m;
        fname << ".csv";
        Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
        csvout.precision(16);
        csvout.setf(std::ios::scientific, std::ios::floatfield);
        csvout << "x rho_e(x) rho_i(x) sum(x) rho(x)" << endl;

        for (int i = 0; i < nx + nghost + 1; ++i) {
            double x = (i + 0.5) * hx + orig_x;
            csvout << x << " ";
            csvout << host_view_rho_e(i) << " ";
            csvout << host_view_rho_i(i) << " ";
            csvout << host_view_sum(i)   << " ";
            csvout << host_view_rho(i)   << endl;
        }
        ippl::Comm->barrier();
    }

protected:
    size_type totalP_m;
    int nt_m;
    Vector_t<int, Dim> nr_m;
    double lbt_m;
    std::string solver_m;
    std::string stepMethod_m;
    std::string directory_m;
    std::vector<std::string> preconditioner_params_m;

    double time_m;
    double dt_m;
    int it_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> hr_m;
    double Q_m;
    Vector_t<double, Dim> origin_m;
    bool isFirstRepartition_m;
    ippl::NDIndex<Dim> domain_m;
    std::array<bool, Dim> decomp_m;
    double rhoNorm_m;
    Vector_t<T, 3> Bext_m;
    T phiWall_m;
};
#endif
