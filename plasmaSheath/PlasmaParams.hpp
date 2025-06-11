// vim: ts=4 sw=4 et
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <algorithm>
#include <limits>
#include <ostream>
#include <string>

#ifndef PLASMA_INPUT_H_
#define PLASMA_INPUT_H_

constexpr double pi       = 3.14159265358979323846;
constexpr double infinity = std::numeric_limits<double>::infinity();

class PlasmaSheathParams {
public:
    // normalization:
    // q is in units of e
    // m is in units of m_i
    // v is in units of v_th_i = sqrt(T_i / m_i)
    // T is in units of T_e
    // n is in units of n_e(x=MPE)
    // x is in units of L_ref, usually the Debye length sqrt(eps T_e / (e² n_e))
    // phi is in units of T_e/e

    // ion charge
    const double Z_i;
    // electron charge (as defined by normalization)
    const double Z_e = -1.0;

    // ion density at MPE, such that Z n_i - n_e = 0
    const double n_i0;
    // electron density at MPE (as defined by normalization)
    const double n_e0 = 1.0;

    // ion mass (as defined by normalization)
    const double m_i = 1.0;
    // electron mass (e.g. 1.0/1836.0)
    const double m_e;

    // ion-electron temperature ratio, τ = T_i / T_e
    const double tau;
    // perp-parallel temperature anisotropy, ν = v_th_perp_i / v_th_par_i
    const double nu;

    // Debye length (as defined by normalization)
    const double D_D = 1.0;
    // ion thermal gyroradius ρ_th_i, in units of L_ref.
    // To have B = 0, set D_C = ∞
    // constexpr double D_C = 10.0;
    const double D_C = infinity;

    // magnetic field incidence angle
    // set this to 90deg for Debye sheath simulations, so that vpar = -vx !
    const double alpha;

    // wall bias. note that phi(x=MPE) = 0
    const double phi0;

    // toggles between adiabatic electrons and kinetic electrons
    const bool kinetic_electrons;

    // derived quantities from the physical parameters
    // in normalized units, v_th_i = 1.0   and v_th_e = √(T_i/tau) √(m_i/m_e) 1/√m_i = 1 / √(τ ~m_e) v_th_i
    //                      ρ_th_i = D_C   and ρ_th_e = Z √~m_e / √τ D_C
    //                      Ω_ci = 1/D_C   and Ω_ce = 1/(Z ~m_e D_C)
    // ion thermal velocity, by definition of normalization
    const double v_th_i = 1.0;
    const double v_th_e; // = 1.0 / Kokkos::sqrt(tau * m_e);  // can't use constexpr since sqrt not constexpr
    const double rho_th_i; // = D_C;
    const double rho_th_e; // = D_C * Z_i * Kokkos::sqrt(m_e / tau);  // can't use constexpr since sqrt not constexpr
    const double Omega_ci; // = 1.0 / D_C;
    const double Omega_ce; // = 1.0 / (Z_i * m_e * D_C);

    // -- GRID PARAMS
    // length of the simulation domain, in units of L_ref
    const double L;
    // resolution of the smallest length scale min(ρ_th_e, λ_D, ρ_th_i).
    const double f_x;
    // resolution of the smallest time scale 2π/Ω_ce. should be < 1.0
    const double f_t;
    // β_max = v_max Δx/Δt, should be < 1.0
    const double CFL_max;

    // rough estimate of the velocity of the ions as the impact the wall, relative to initial v_x
    const double f_ion_speedup;
    // safety factor in units of the species' thermal velocity, for v_max calculation
    // (e.g. typical sampled v_par is v_th_i, but some ions may get sampled with v_par = 6 v_th_i)
    const double f_v_th_safety;
    // maximum velocity expected to be encountered in the simulation
    const double v_max;  // = max(...)

    // only accept ions with 0 < -v_x < v_trunc_i
    const double v_trunc_i; // = v_max / f_ion_speedup;
    // only accept ions with 0 < -v_x < v_trunc_e
    const double v_trunc_e; // = v_max;

    // postprocessing of simulation parameters
    // resolution such that dx << smallest length scale
    // can't use constexpr since rho_th_e, min(), ceil() not marked as constexpr
    const double dx0; // = f_x * std::min({kinetic_electrons ? rho_th_e : infinity, rho_th_i, D_D});
    const unsigned int nx; // = Kokkos::ceil(L / dx0);
    // the actual dx
    const double dx; // = L / (double)nx;
    // timestep as imposed by CFL or cyclotron frequency
    const double dt; // min(...)

    // -- SIMULATION PARAMS
    const unsigned int num_particles;
    const unsigned int num_timesteps;
    // load balancer type ?
    const double lbt;

    // -- SOLVER PARAMS
    const std::string solver;
    const std::string step_method;

    // -- OUTPUT PARAMS --
    const std::string directory;
    // dump once every <dump_interval> steps
    const unsigned int dump_interval_plasma;
    const unsigned int dump_interval_particles;

    PlasmaSheathParams(
        double Z_i_, double n_i0_, double m_e_,
        double tau_, double nu_,
        double D_C_,
        double alpha_rad_,
        double phi0_,
        bool kinetic_electrons_,
        double L_, double f_x_, double f_t_, double CFL_max_,
        double f_ion_speedup_, double f_v_th_safety_,
        unsigned int dump_interval_plasma_, unsigned int dump_interval_particles_,
        unsigned int num_particles_, unsigned int num_timesteps_,
        double lbt_, std::string solver_, std::string step_method_, std::string directory_
    ) : Z_i(Z_i_),
        n_i0(n_i0_),
        m_e(m_e_),
        tau(tau_),
        nu(nu_),
        D_C(D_C_),
        alpha(alpha_rad_),
        phi0(phi0_),
        kinetic_electrons(kinetic_electrons_),
        L(L_),
        f_x(f_x_),
        f_t(f_t_),
        CFL_max(CFL_max_),
        f_ion_speedup(f_ion_speedup_),
        f_v_th_safety(f_v_th_safety_),

        v_th_e(1.0 / Kokkos::sqrt(tau * m_e)),
        rho_th_i(D_C),
        rho_th_e(D_C * Z_i * Kokkos::sqrt(m_e / tau)),
        Omega_ci(1.0 / D_C),
        Omega_ce(1.0 / (Z_i * m_e * D_C)),
        v_max(std::max({
            // ions that get sampled with some velocity get accelerated towards the wall
            f_ion_speedup * f_v_th_safety * v_th_i,
            // electrons are reflected, so their max velocity is the one they're sampled with
            (kinetic_electrons ? f_v_th_safety * v_th_e : 0.0),
        })),
        v_trunc_i(v_max / f_ion_speedup),
        v_trunc_e(v_max),
        dx0(f_x * std::min({kinetic_electrons ? rho_th_e : infinity, rho_th_i, D_D})),
        nx(Kokkos::ceil(L / dx0)),
        dx(L / (double)nx),
        dt(std::min({
            // only resolve the cyclotron frequency if B > 0, i.e. D_C < oo
            std::isfinite(D_C) ? (f_t * 2.0 * pi / std::max({Omega_ci, kinetic_electrons ? Omega_ce
                                                                                         : 0.0}))
                               : infinity,
            // time step constraint due to the CFL condition
            dx / v_max * CFL_max
        })),

        num_particles(num_particles_),
        num_timesteps(num_timesteps_),
        lbt(lbt_),
        solver(solver_),
        step_method(step_method_),
        directory(directory_),
        dump_interval_plasma(dump_interval_plasma_),
        dump_interval_particles(dump_interval_particles_)
    {}

};


// TODO: main function so that you can run just the parsing of the input file and see the parameters
#endif
