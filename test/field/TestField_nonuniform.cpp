// Tests the application of various kinds of boundary conditions on fields
#include "Ippl.h"

#include <array>
#include <iostream>
#include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t      = ippl::Cartesian<double, dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;
        typedef ippl::BConds<field_type, dim> bc_type;

        int pt = 8;
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);

        std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        double dx                        = 1.0 / double(pt);
        ippl::Vector<double, dim> hx     = dx;
        ippl::Vector<double, dim> origin = 0;

        // some prints
        std::cout << "dx = " << dx << std::endl;
        std::cout << layout << std::endl;

        using exec_space  = typename Kokkos::View<double*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        ippl::Vector<Kokkos::View<double*>, dim> spacings;
        for (unsigned int d = 0; d < dim; ++d) {
            spacings[d] = Kokkos::View<double*>("spacingD", owned[d].length());
            Kokkos::parallel_for("set_spacing", policy_type(0, spacings[d].extent(0)), 
                KOKKOS_LAMBDA(int i) {
                    if (i < 4 ) {
                        spacings[d](i) = dx;
                    } else {
                        spacings[d](i) = dx/2;
                    }
                });
        }

        Mesh_t mesh(spacings, origin);

        field_type field(mesh, layout, 1);

        // Periodic BC
        bc_type bcField;
        for (unsigned int i = 0; i < 6; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
        }
        bcField.apply(field);

        field = 1.0;

        const ippl::NDIndex<dim>& lDom       = layout.getLocalNDIndex();
        const int nghost                     = field.getNghost();
        typename field_type::view_type& view = field.getView();
        auto& meshSpacing                    = mesh.getMeshSpacing();

        Kokkos::parallel_for(
            "Assign field", field.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // local to global index conversion
                const size_t ig = i + lDom[0].first() - nghost;
                const size_t jg = j + lDom[1].first() - nghost;
                const size_t kg = k + lDom[2].first() - nghost;

                double hx = meshSpacing[0](i);
                double hy = meshSpacing[1](j);
                double hz = meshSpacing[2](k);

                double x = (ig + 0.5) * hx + origin[0];
                double y = (jg + 0.5) * hy + origin[1];
                double z = (kg + 0.5) * hz + origin[2];

                std::cout << "i = " << ig << ", j = " << jg << ", k = " << kg << std::endl;
                std::cout << "spacing = " << hx << "," << hy << "," << hz << std::endl;
            });
    }
    ippl::finalize();

    return 0;
}
