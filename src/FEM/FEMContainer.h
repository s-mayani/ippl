// Class FEMContainer
// This class holds a collection of DOFs (degrees of freedom) for a finite element mesh.
// The DOFs are stored on multiple ippl::Fields, split by different entity types (vertices, edges in
// x/y/z, faces in xy/xz/yz, etc.). This allows for easy boundary condition application and field
// operations.

#ifndef IPPL_FEMCONTAINER_H
#define IPPL_FEMCONTAINER_H

#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "Types/ViewTypes.h"

#include "DOFArray.h"
#include "FEM/Entity.h"
#include "FEMHelperStructs.h"
#include "Field/HaloCells.h"
#include "Kokkos_Macros.hpp"

namespace ippl {

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    class FEMContainer {
    public:
        // check that EntityTypes and DOFNums are tuples
        static_assert(is_tuple_v<EntityTypes>, "EntityTypes must be a std::tuple");
        static_assert(is_tuple_v<DOFNums>, "DOFNums must be a std::tuple");
        // Check, that EntityTypes and DOFNums have same size
        static_assert(std::tuple_size_v<EntityTypes> == std::tuple_size_v<DOFNums>,
                      "Number of EntityTypes must match number of DOFNums");
        // Check that all EntityTypes are unique
        static_assert(!tuple_has_duplicates_v<EntityTypes>, "EntityTypes must be unique types");
        // Check that all EntityTypes are derived from Entity
        // Check that all EntityTypes are derived from Entity with matching dimension
        static_assert(
            [] {
                bool all_derived = true;
                std::apply(
                    [&all_derived](auto... entity_types) {
                        ((all_derived =
                              all_derived
                              && std::is_base_of_v<ippl::Entity<decltype(entity_types), Dim>,
                                                   decltype(entity_types)>),
                         ...);
                    },
                    EntityTypes{});
                return all_derived;
            }(),
            "All EntityTypes must be derived from Entity");

        static constexpr unsigned dim = Dim;
        using value_type              = T;

        using Mesh_t      = UniformCartesian<T, Dim>;
        using Layout_t    = FieldLayout<Dim>;
        using Centering_t = Cell;  // This is only a placeholder since the template does nothing
                                   // with it at the moment Actual centering is defined by the
                                   // EntityTypes and done using the SubFieldLayout class

        // Build tuple of field types
        using FieldTuple = typename FieldTupleBuilder<T, Dim, Mesh_t, EntityTypes, DOFNums>::type;
        using ViewTuple =
            typename FieldTupleBuilder<T, Dim, Mesh_t, EntityTypes, DOFNums>::view_type;

        static constexpr unsigned NEntitys =
            std::tuple_size_v<EntityTypes>;  // Number of entity types with DOFs

        FEMContainer();
        FEMContainer(Mesh_t& m, const Layout_t& l, int nghost = 1);
        FEMContainer(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other);

        void initialize(Mesh_t& m, const Layout_t& l, int nghost = 1);

        FEMContainer<T, Dim, EntityTypes, DOFNums> deepCopy() const;

        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator=(T value);
        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator=(
            const FEMContainer<T, Dim, EntityTypes, DOFNums>& other);

        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator+=(T value);
        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator+=(
            const FEMContainer<T, Dim, EntityTypes, DOFNums>& other);

        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator-=(T value);
        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator-=(
            const FEMContainer<T, Dim, EntityTypes, DOFNums>& other);

        FEMContainer<T, Dim, EntityTypes, DOFNums> operator+(
            const FEMContainer<T, Dim, EntityTypes, DOFNums>& other) const;
        FEMContainer<T, Dim, EntityTypes, DOFNums> operator-(
            const FEMContainer<T, Dim, EntityTypes, DOFNums>& other) const;

        FEMContainer<T, Dim, EntityTypes, DOFNums> operator+(T scalar) const;
        FEMContainer<T, Dim, EntityTypes, DOFNums> operator-(T scalar) const;
        FEMContainer<T, Dim, EntityTypes, DOFNums> operator*(T scalar) const;
        FEMContainer<T, Dim, EntityTypes, DOFNums> operator/(T scalar) const;

        T max() const;
        T min() const;

        // Friend function for scalar * FEMContainer
        friend FEMContainer<T, Dim, EntityTypes, DOFNums> operator*(
            T scalar, const FEMContainer<T, Dim, EntityTypes, DOFNums>& container) {
            return container * scalar;
        }

        // Friend function for norm (callable as ippl::norm via ADL)
        friend T norm(const FEMContainer<T, Dim, EntityTypes, DOFNums>& container, int p = 2) {
            return container.norm(p);
        }

        template <typename TA, typename TB>
        struct innerProduct_functor {
            innerProduct_functor(TA view_a, TB view_b, auto numDOFs)
                : view_a_{view_a}
                , view_b_{view_b}
                , numDOFs_(numDOFs) {}

            template <typename ARGS, typename TV>
            KOKKOS_INLINE_FUNCTION void operator()(const ARGS& args, TV& val) const {
                // Compute dot product of DOFArrays at this position
                auto dof_a = apply(view_a_, args);
                auto dof_b = apply(view_b_, args);
                for (unsigned dof = 0; dof < numDOFs_; ++dof) {
                    val += dof_a[dof] * dof_b[dof];
                }
            }

            KOKKOS_INLINE_FUNCTION void join(auto& dst, const auto& src) const { dst += src; }

            KOKKOS_INLINE_FUNCTION void init(auto& dst) const { dst = 0; }

        private:
            TA view_a_;
            TB view_b_;
            const unsigned numDOFs_;
        };

        // Friend function for inner product (callable as ippl::innerProduct via ADL)
        friend T innerProduct(const FEMContainer<T, Dim, EntityTypes, DOFNums>& a,
                              const FEMContainer<T, Dim, EntityTypes, DOFNums>& b) {
            T localSum = 0.0;

            // Compute inner product over all fields in the tuple
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                (([&]() {
                     // FIXME: GPU error: An extended __host__ __device__ lambda cannot be defined
                     // inside a generic lambda expression("operator()").
                     const auto& field_a = std::get<Is>(a.data_m);
                     const auto& field_b = std::get<Is>(b.data_m);
                     auto view_a         = field_a.getView();
                     auto view_b         = field_b.getView();

                     // Compute local inner product by iterating over view
                     T fieldLocalSum            = 0.0;
                     constexpr unsigned numDOFs = numDOFs_m[Is];

                     innerProduct_functor lambda(view_a, view_b, numDOFs);

                     ippl::parallel_reduce("FEMContainer innerProduct field",
                                           field_a.getFieldRangePolicy(), lambda,
                                           Kokkos::Sum<T>(fieldLocalSum));

                     localSum += fieldLocalSum;
                 }()),
                 ...);
            }(std::make_index_sequence<NEntitys>{});

            // Global sum across MPI ranks
            T globalSum = 0.0;
            ippl::Comm->allreduce(localSum, globalSum, 1, std::plus<T>());
            return globalSum;
        }

        // TODO implement inf norm
        T norm(int p = 2) const;

        T getVolumeAverage() const;

        void fillHalo();
        void accumulateHalo();
        void accumulateHalo_noghost(int nghost = 1);

        int getNghost() const { return nghost_m; }

        unsigned int getNumFields() { return NEntitys; }

        KOKKOS_INLINE_FUNCTION Mesh_t& get_mesh() const { return *mesh_m; }

        KOKKOS_INLINE_FUNCTION Layout_t& getLayout() { return *VertexLayout_m; }
        KOKKOS_INLINE_FUNCTION const Layout_t& getLayout() const { return *VertexLayout_m; }

        // Access individual layouts
        template <typename EntityType>
        Layout_t& getLayout() {
            constexpr unsigned index = TagIndex<EntityTypes>::template index<EntityType>();
            return std::get<index>(layout_m);
        }

        // Const access individual layouts
        template <typename EntityType>
        const Layout_t& getLayout() const {
            constexpr unsigned index = TagIndex<EntityTypes>::template index<EntityType>();
            return std::get<index>(layout_m);
        }

        // Get the number of DOFs for each entity type
        template <typename EntityType>
        unsigned getNumDOFs() const {
            constexpr unsigned index = TagIndex<EntityTypes>::template index<EntityType>();
            return numDOFs_m[index];
        }

        template <typename EntityType>
        bool hasView() const {
            // check if EntityType is in EntityTypes
            return TagIndex<EntityTypes>::template contains<EntityType>();
        }

        template <typename EntityType>
        const decltype(std::tuple_element_t<TagIndex<EntityTypes>::template index<EntityType>(),
                                            FieldTuple>())::view_type&
        getView() const {
            // Get index of EntityType in EntityTypes
            constexpr unsigned index = TagIndex<EntityTypes>::template index<EntityType>();

            return std::get<index>(data_m).getView();
        }

        /**
         * @brief Get a range policy for iterating over a specific entity type field
         *
         * @tparam EntityType The entity type to get the range policy for
         * @tparam PolicyArgs Additional template parameters for the range policy
         * @param nghost Number of ghost layers to include in the range policy (default 0)
         * @return Range policy for iterating over the specified entity type field
         */
        template <typename EntityType, class... PolicyArgs>
        auto getFieldRangePolicy(const int nghost = 1) const {
            constexpr unsigned index = TagIndex<EntityTypes>::template index<EntityType>();
            return std::get<index>(data_m).template getFieldRangePolicy<PolicyArgs...>(nghost);
        }

        const ViewTuple getAllViews() const {
            ViewTuple views;
            for (unsigned i = 0; i < NEntitys; ++i) {
                views[i] = std::get<i>(data_m).getView();
            }
            return views;
        }

        constexpr static auto getEntityTypes() { return EntityTypes{}; }

        /**
         * @brief Set boundary conditions for all entity type fields
         *
         * Takes an array specifying which BC type to apply on each boundary face
         * and applies corresponding boundary conditions to each field in the container.
         *
         * Only allowed for BC types that do not require constant values, i.e., PERIODIC_FACE,
         * ZERO_FACE, NO_FACE.
         *
         * @param bcTypes Array of boundary condition types for each face (0 to 2*Dim-1)
         */
        void setFieldBC(const std::array<FieldBC, 2 * Dim>& bcTypes);

        /**
         * @brief Set boundary conditions for all entity type fields
         *
         * Takes an array specifying which BC type to apply on each boundary face
         * and applies corresponding boundary conditions to each field in the container.
         *
         * Allowed for all BC types, contant values needed for CONSTANT_FACE and EXTRAPOLATE_FACE
         * given in bcValues and bcSlopes arrays.
         *
         * @param bcTypes Array of boundary condition types for each face (0 to 2*Dim-1)
         */
        void setFieldBC(const std::array<FieldBC, 2 * Dim>& bcTypes,
                        std::array<T, 2 * Dim> bcValues, std::array<T, 2 * Dim> bcSlopes);

        /**
         * @brief Set boundary conditions from a BConds object
         *
         * Extracts BC types and constant values from a BConds object and applies them
         * to all fields in the FEMContainer. This is useful for setting CONSTANT_FACE BCs
         * with specific values.
         *
         * @tparam Field Any field type with compatible BConds
         * @param bconds BConds object containing boundary conditions with values
         */
        template <typename Field>
        void setFieldBC(const BConds<Field, Dim>& bconds);

        /**
         * @brief Get the boundary condition types for all faces
         *
         * Returns a reference to the array of boundary condition types for each face.
         *
         * @return Const reference to array of boundary condition types
         */
        const std::array<FieldBC, 2 * Dim>& getFieldBCTypes() const { return bcTypes_m; }

        /**
         * @brief Get boundary conditions as BConds object
         *
         * Constructs and returns a BConds object containing the boundary conditions.
         * The BConds is templated on a simple scalar Field type for compatibility with solvers.
         *
         * @return BConds object with all boundary conditions and values
         */
        BConds<Field<T, Dim, Mesh_t, Cell>, Dim> getFieldBC() const;

        /**
         * @brief Apply boundary conditions to all fields
         *
         * Calls apply() on the boundary conditions for each field in the container.
         * This enforces the boundary conditions on the field values.
         */
        void applyBC();

        /**
         * @brief Assign ghost to physical cells for all fields
         *
         * Calls assignGhostToPhysical() on the boundary conditions for each field
         * in the container. This is used for periodic boundary conditions to copy
         * values from ghost cells back to physical cells.
         */
        void assignGhostToPhysical();

    private:
        // Helperfunction to create numDOFs_m array
        template <unsigned... DOFNum>
        static constexpr std::array<unsigned, NEntitys> createNumDOFsArray(
            std::tuple<std::integral_constant<unsigned, DOFNum>...>) {
            return std::array<unsigned, NEntitys>{DOFNum...};
        }

        static constexpr std::array<unsigned, NEntitys> numDOFs_m =
            createNumDOFsArray(DOFNums{});  // Number of DOFs for each field

        FieldTuple data_m;  // Fields with DOFs for each entity type with DOFs

        std::array<SubFieldLayout<Dim>, NEntitys>
            layout_m;  // Layouts for each entity type with DOFs

        int nghost_m;

        Mesh_t* mesh_m;  // Pointer to the mesh

        Layout_t* VertexLayout_m;  // Layout of vertices

        std::array<FieldBC, 2 * Dim> bcTypes_m;  // Boundary condition types for each face
        std::array<T, 2 * Dim>
            bcValues_m{};  // Offset values for CONSTANT_FACE and EXTRAPOLATE_FACE BCs
        std::array<T, 2 * Dim> bcSlopes_m{};  // Slope values for EXTRAPOLATE_FACE BCs (default 1.0)
    };
}  // namespace ippl

#include "FEMContainer.hpp"

#endif  // IPPL_FEMCONTAINER_H
