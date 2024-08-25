#include "Ippl.h"

#include <array>
#include <iostream>
#include <random>
#include <typeinfo>
#include "Utility/ParameterList.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS

KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z, double sigma = 0.05, double mu = 0.5) {
    double pi        = Kokkos::numbers::pi_v<double>;
    double prefactor = (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return prefactor * exp(-r2 / (2 * sigma * sigma));
}

constexpr unsigned int dim = 3;
using Mesh_t               = ippl::UniformCartesian<double, dim>;
using Centering_t          = Mesh_t::DefaultCentering;
typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> CxField_t;

void dumpField(CxField_t& field, std::string what) {

    Inform m("FSopen::dumpScalField() ");

    if (ippl::Comm->size() > 1) {
        return;
    }
    m << "*** START DUMPING SCALAR FIELD ***" << endl;

    int step = 0;

    std::string type;
    std::string unit;
    bool isVectorField = false;

    auto localIdx = field.getOwned();
    auto mesh_mp  = &(field.get_mesh());
    auto spacing  = mesh_mp->getMeshSpacing();
    auto origin   = mesh_mp->getOrigin();
    
    auto fieldV      = field.getView();
    auto field_hostV = field.getHostMirror();

    Kokkos::deep_copy(field_hostV, fieldV);     

    boost::filesystem::path file("../input-files/");
    boost::format filename("%1%-%2%-%|3$06|.dat");
    std::string basename = "fieldDump";
    filename % basename % (what + std::string("_") + type) % step;
    file /= filename.str();
    m << "*** FILE NAME " + file.string() << endl;
    std::ofstream fout(file.string(), std::ios::out);

    unit = "?";

    fout << std::setprecision(9);

    fout << "# " << what << " " << type << " data on grid" << std::endl
         << "# origin= " << std::fixed << origin << " h= " << std::fixed << spacing << std::endl 
         << std::setw(5)  << "i"
         << std::setw(5)  << "j"
         << std::setw(5)  << "k"
         << std::setw(17) << "x [m]"
         << std::setw(17) << "y [m]"
         << std::setw(17) << "z [m]";
    
    if (isVectorField) {
        fout << std::setw(10) << what << "x [" << unit << "]"
             << std::setw(10) << what << "y [" << unit << "]"
             << std::setw(10) << what << "z [" << unit << "]";
    } else {
        fout << std::setw(13) << what << " [" << unit << "]";
    }

    fout << std::endl;

    for (int i = localIdx[0].first(); i <= localIdx[0].last(); i++) {
        for (int j = localIdx[1].first(); j <= localIdx[1].last(); j++) {
            for (int k = localIdx[2].first(); k <= localIdx[2].last(); k++) {
                
                // define the physical points (cell-centered)
                const double x = i * spacing[0] + origin[0];        
                const double y = j * spacing[1] + origin[1];        
                const double z = k * spacing[2] + origin[2];     

                const double a = field_hostV(i,j,k).real();
                const double b = field_hostV(i,j,k).imag();
                const double c = (a*a) + (b*b);
                
                fout << std::setw(5) << i + 1
                     << std::setw(5) << j + 1
                     << std::setw(5) << k + 1
                     << std::setw(17) << x
                     << std::setw(17) << y
                     << std::setw(17) << z
                     << std::scientific << "\t" << c
                     << std::endl;
            }
        }
    }
    fout.close();
    m << "*** FINISHED DUMPING " + what + " FIELD ***" << endl;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        std::array<int, dim> pt = {64, 64, 64};
        ippl::Index I(pt[0]);
        ippl::Index J(pt[1]);
        ippl::Index K(pt[2]);
        ippl::NDIndex<dim> owned(I, J, K);

        std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        std::array<double, dim> dx = {1.0, 1.0, 1.0};
        ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
        ippl::Vector<double, 3> origin = {0, 0, 0};
        Mesh_t mesh(owned, hx, origin);

        CxField_t rho(mesh, layout);

        ippl::ParameterList fftParams;

        fftParams.add("use_heffte_defaults", true);

        typedef ippl::FFT<ippl::CCTransform, CxField_t> FFT_type;

        std::unique_ptr<FFT_type> fft;

        fft = std::make_unique<FFT_type>(layout, fftParams);

        // assign the rho field with a gaussian
        auto view_rho    = rho.getView();
        const int nghost = rho.getNghost();
        const auto& ldom = layout.getLocalNDIndex();

        Kokkos::parallel_for(
            "Assign rho field", rho.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local to global indices
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                // define the physical points (cell-centered)
                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                view_rho(i, j, k).real() = gaussian(x, y, z, 4, 32);
            });

        dumpField(rho, "rho");

        // Forward transform
        fft->transform(ippl::BACKWARD, rho);

        dumpField(rho, "rhotr");

        // Reverse transform
        fft->transform(ippl::FORWARD, rho);

        dumpField(rho, "rhoinv");
    }
    ippl::finalize();

    return 0;
}
