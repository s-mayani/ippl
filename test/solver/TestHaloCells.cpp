#include "Ippl.h"

#include "Meshes/Centering.h"
#include "PoissonSolvers/FEMPoissonSolver.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {

        Inform m("");
        Inform msg2all("", INFORM_ALL_NODES);
        m << "Starting up Test Ghost cells fillHalo..." << endl;

        using T = double;
        const unsigned int Dim = 2;

        using Mesh_t   = ippl::UniformCartesian<T, Dim>;
        using Field_t  = ippl::Field<T, Dim, Mesh_t, Cell>;
        using BConds_t = ippl::BConds<Field_t, Dim>;

        int me = ippl::Comm->rank();

        // Domain: [-1, 1]
        const int numNodesPerDim = 4;
        double domain_start      = -1.0;
        double domain_end        = 1.0;
        int numCellsPerDim       = numNodesPerDim - 1; 
        int numGhosts            = 1;

        const ippl::Vector<unsigned, Dim> nodesPerDimVec(numNodesPerDim);
        ippl::NDIndex<Dim> domain(nodesPerDimVec);
        ippl::Vector<T, Dim> cellSpacing((domain_end - domain_start) / static_cast<T>(numCellsPerDim));
        ippl::Vector<T, Dim> origin(domain_start);
        Mesh_t mesh(domain, cellSpacing, origin);

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, domain, isParallel);

        auto ldom = layout.getLocalNDIndex();
        msg2all << "ID: " << me << ", local dom = " << ldom << endl;

        Field_t rhs(mesh, layout, numGhosts);

        m << "Field has been defined" << endl;

        // Define boundary conditions
        BConds_t bcField;
        for (unsigned int i = 0; i < 2 * Dim; ++i) {
            bcField[i] = std::make_shared<ippl::ZeroFace<Field_t>>(i);
        }
        rhs.setFieldBC(bcField);

        m << "BCs set" << endl;

        rhs.fillHalo();

        m << "After fillHalo" << endl;

    }
    ippl::finalize();

    return 0;
}
