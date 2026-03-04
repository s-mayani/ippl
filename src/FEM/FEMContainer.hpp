// Class FEMContainer
// This class holds a collection of DOFs (degrees of freedom) for a finite element mesh.
// The DOFs are stored in a multi-dimensional allay, split by their dimension (vertices, edges, faces, etc.).
// This allows for easy boundary condition application and field operations.

namespace ippl {

    // Functor for FEMContainer norm computation
    // Must be defined at namespace scope for CUDA compatibility
    template <typename ViewType, typename T, std::size_t NumDOFs, typename IndexArrayType>
    struct FEMContainerNormFunctor {
        ViewType view_;
        int p_;

        KOKKOS_INLINE_FUNCTION
        void operator()(const IndexArrayType& args, T& val) const {
            auto dofArray = apply(view_, args);
            // Sum over all DOF elements in this DOFArray
            for (std::size_t dof = 0; dof < NumDOFs; ++dof) {
                val += std::pow(Kokkos::abs(dofArray[dof]), p_);
            }
        }
    };

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>::FEMContainer() {}

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>::FEMContainer(Mesh_t& m, const Layout_t& l, int nghost) {
        initialize(m, l, nghost);
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>::FEMContainer(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other)
        : nghost_m(other.nghost_m)
        , mesh_m(other.mesh_m)
        , VertexLayout_m(other.VertexLayout_m)
        , bcTypes_m(other.bcTypes_m) {

        // Copy the layout array
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((layout_m[Is] = other.layout_m[Is]), ...);
        }(std::make_index_sequence<NEntitys>{});

        // Copy the data tuple (fields)
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((std::get<Is>(data_m) = std::get<Is>(other.data_m).deepCopy()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums> FEMContainer<T, Dim, EntityTypes, DOFNums>::deepCopy() const {
        // Use the copy constructor which already does deep copying
        return FEMContainer<T, Dim, EntityTypes, DOFNums>(*this);
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::initialize(Mesh_t& m, const Layout_t& l, int nghost) {

        // Set mesh and ghostcells
        nghost_m = nghost;
        mesh_m = &m;

        // Initialize layout pointer to point to the input layout
        // const_cast is needed because we store a non-const pointer to allow
        // preconditioners to work with both Field and FEMContainer types
        VertexLayout_m = const_cast<Layout_t*>(&l);

        // Initialize boundary condition types to NO_FACE by default
        bcTypes_m.fill(NO_FACE);

        // Get domain and communicator of the layout
        NDIndex<Dim> domain = l.getDomain();

        mpi::Communicator comm = l.comm;
        auto parallel = l.isParallel();

        // loop trough all the entity types
        [&]<std::size_t... i>(std::index_sequence<i...>) {
            // Fold over all indices
            (([&]() {                
                // Get the dimension of this entity type
                std::array<bool, Dim> dir = std::tuple_element_t<i, EntityTypes>{}.getDir();
                
                // Create local subdomain
                NDIndex<Dim> subDomain = domain;
                
                for (unsigned int d = 0; d < Dim; ++d) {
                    subDomain[d] = dir[d] ? subDomain[d].cut(1) : subDomain[d];
                }
                
                // Create a sub field layout for this element type and direction
                layout_m[i].comm = comm;
                layout_m[i].initialize(domain, subDomain, parallel, l.isAllPeriodic_m);
                
                // Initialize field for this element type and direction
                std::get<i>(data_m).initialize(*mesh_m, layout_m[i], nghost);
            })(), ...);
        }(std::make_index_sequence<std::tuple_size_v<EntityTypes>>{});
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>& FEMContainer<T, Dim, EntityTypes, DOFNums>::operator=(T value) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                std::get<Is>(data_m) = value;
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});
       
        return *this;
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>& FEMContainer<T, Dim, EntityTypes, DOFNums>::operator+=(T value) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                std::get<Is>(data_m) = std::get<Is>(data_m) + value;
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});

        return *this;
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>& FEMContainer<T, Dim, EntityTypes, DOFNums>::operator+=(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                std::get<Is>(data_m) = std::get<Is>(data_m) + std::get<Is>(other.data_m);
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});

        return *this;
    }
    
        template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
        FEMContainer<T, Dim, EntityTypes, DOFNums>& FEMContainer<T, Dim, EntityTypes, DOFNums>::operator-=(T value) {
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                (([&]() {
                    std::get<Is>(data_m) = std::get<Is>(data_m) - value;
                }()), ...);
            }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});
    
            return *this;
        }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>& FEMContainer<T, Dim, EntityTypes, DOFNums>::operator-=(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                std::get<Is>(data_m) = std::get<Is>(data_m) - std::get<Is>(other.data_m);
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});

        return *this;
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums> FEMContainer<T, Dim, EntityTypes, DOFNums>::operator+(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other) const {
        FEMContainer<T, Dim, EntityTypes, DOFNums> result(*this);
        result += other;
        return result;
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums> FEMContainer<T, Dim, EntityTypes, DOFNums>::operator-(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other) const {
        FEMContainer<T, Dim, EntityTypes, DOFNums> result(*this);
        result -= other;
        return result;
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums> FEMContainer<T, Dim, EntityTypes, DOFNums>::operator*(T scalar) const {
        FEMContainer<T, Dim, EntityTypes, DOFNums> result(*this);
        // Multiply each field in the tuple by the scalar
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                std::get<Is>(result.data_m) = std::get<Is>(data_m) * scalar;
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});
        return result;
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums> FEMContainer<T, Dim, EntityTypes, DOFNums>::operator-(T scalar) const {
        FEMContainer<T, Dim, EntityTypes, DOFNums> result(*this);
        // Subtract scalar from each field in the tuple
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                std::get<Is>(result.data_m) = std::get<Is>(data_m) - scalar;
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});
        return result;
    }


    // TODO: Only for testing purposes, not efficient
    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    T FEMContainer<T, Dim, EntityTypes, DOFNums>::max() const {
        T maxVal = std::numeric_limits<T>::lowest();
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                auto fieldMax = std::get<Is>(data_m).max();
                for (unsigned int i = 0; i < fieldMax.data.size(); ++i) {
                    if (fieldMax[i] > maxVal) {
                        maxVal = fieldMax[i];
                    }
                }
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});
        return maxVal;
    }

    // TODO:Only for testing purposes, not efficient
    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    T FEMContainer<T, Dim, EntityTypes, DOFNums>::min() const {
        T minVal = std::numeric_limits<T>::max();
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                auto fieldMin = std::get<Is>(data_m).min();
                for (unsigned int i = 0; i < fieldMin.data.size(); ++i) {
                    if (fieldMin[i] < minVal) {
                        minVal = fieldMin[i];
                    }
                }
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});
        return minVal;
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    T FEMContainer<T, Dim, EntityTypes, DOFNums>::norm(int p) const {
        T total_sum = 0;

        // Iterate over all fields in the container
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                constexpr std::size_t numDOFs = std::get<Is>(DOFNums{}).value;
                const auto& field = std::get<Is>(data_m);
                auto layout = field.getLayout();
                auto view = field.getView();

                using exec_space = typename std::remove_reference_t<decltype(field)>::execution_space;
                using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;

                T local_sum = 0;

                // Use the functor defined at namespace scope
                using functor_t = FEMContainerNormFunctor<decltype(view), T, numDOFs, index_array_type>;

                // Compute sum of |element|^p for all DOF elements
                ippl::parallel_reduce(
                    "FEMContainer::norm", field.getFieldRangePolicy(),
                    functor_t{view, p},
                    Kokkos::Sum<T>(local_sum));

                // Reduce across MPI ranks
                T global_sum = 0;
                layout.comm.allreduce(local_sum, global_sum, 1, std::plus<T>());

                total_sum += global_sum;
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});

        return std::pow(total_sum, 1.0 / p);
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>& FEMContainer<T, Dim, EntityTypes, DOFNums>::operator=(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other) {
        // Copy member variables
        nghost_m = other.nghost_m;
        mesh_m = other.mesh_m;
        VertexLayout_m = other.VertexLayout_m;
        bcTypes_m = other.bcTypes_m;

        // Copy the layout array
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((layout_m[Is] = other.layout_m[Is]), ...);
        }(std::make_index_sequence<NEntitys>{});

        // Copy the data tuple (fields)
        // We need to create new fields using our copied layouts, not the other's layouts
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                // Create a temporary field using our own layout
                auto temp = std::get<Is>(other.data_m).deepCopy();
                // Reinitialize it with our copied layout
                std::get<Is>(data_m).initialize(*mesh_m, layout_m[Is], nghost_m);
                // Copy the data
                Kokkos::deep_copy(std::get<Is>(data_m).getView(), temp.getView());
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});

        return *this;
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::fillHalo() {
        // Apply directional halo exchange for each entity type
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                // Get the direction array for this entity type
                using EntityType = std::tuple_element_t<Is, EntityTypes>;
                constexpr std::array<bool, Dim> entityDir = EntityType{}.getDir();

                // Create exchange direction array: exchange in directions where entity does NOT extend
                std::array<bool, Dim> exchangeDir;
                for (unsigned d = 0; d < Dim; ++d) {
                    exchangeDir[d] = !entityDir[d];
                }

                // Call fillHalo with directional filtering
                std::get<Is>(data_m).fillHalo(exchangeDir);
            }()), ...);
        }(std::make_index_sequence<NEntitys>{});
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::accumulateHalo() {
        // Apply directional halo exchange for each entity type
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                // Get the direction array for this entity type
                using EntityType = std::tuple_element_t<Is, EntityTypes>;
                constexpr std::array<bool, Dim> entityDir = EntityType{}.getDir();

                // Create exchange direction array: exchange in directions where entity does NOT extend
                std::array<bool, Dim> exchangeDir;
                for (unsigned d = 0; d < Dim; ++d) {
                    exchangeDir[d] = !entityDir[d];
                }

                // Call accumulateHalo with directional filtering
                std::get<Is>(data_m).accumulateHalo(exchangeDir);
            }()), ...);
        }(std::make_index_sequence<NEntitys>{});
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::accumulateHalo_noghost(int nghost) {
        // Apply directional halo exchange for each entity type, excluding corner ghost cells
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                // Get the direction array for this entity type
                using EntityType = std::tuple_element_t<Is, EntityTypes>;
                constexpr std::array<bool, Dim> entityDir = EntityType{}.getDir();

                // Create exchange direction array: exchange in directions where entity does NOT extend
                std::array<bool, Dim> exchangeDir;
                for (unsigned d = 0; d < Dim; ++d) {
                    exchangeDir[d] = !entityDir[d];
                }

                // Call accumulateHalo_noghost with directional filtering
                std::get<Is>(data_m).accumulateHalo_noghost(exchangeDir, nghost);
            }()), ...);
        }(std::make_index_sequence<NEntitys>{});
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::setFieldBC(const std::array<FieldBC, 2*Dim>& bcTypes) {
        // Store the boundary condition types
        bcTypes_m = bcTypes;

        // Apply boundary conditions to each field in the container
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                // Get the entity type and its direction
                using EntityType = std::tuple_element_t<Is, EntityTypes>;
                constexpr std::array<bool, Dim> entityDir = EntityType{}.getDir();

                // Get the field type for this entity
                using Field_t = std::tuple_element_t<Is, FieldTuple>;
                using BConds_t = BConds<Field_t, Dim>;
                using value_type = typename Field_t::value_type;

                // Create boundary conditions for this field
                BConds_t bcField;

                // Domain boundary faces are numbered:
                // Face 0 = X lower (X=0 plane), Face 1 = X upper (X=max plane)
                // Face 2 = Y lower (Y=0 plane), Face 3 = Y upper (Y=max plane)
                // Face 4 = Z lower (Z=0 plane), Face 5 = Z upper (Z=max plane)
                //
                // An entity defined on a face F perpendicular to dimension d
                // can have a BC if the entity does NOT vary in dimension d (i.e., entityDir[d] == false)
                //
                // Examples:
                // - Vertex (entityDir = [false, false, false]): BCs on all faces
                // - EdgeX (entityDir = [true, false, false]): BCs on Y and Z faces (faces 2,3,4,5)
                // - FaceXY (entityDir = [true, true, false]): BCs on Z faces (faces 4,5)
                // - Hexahedron (entityDir = [true, true, true]): No BCs

                for (unsigned int face = 0; face < 2 * Dim; ++face) {
                    // Determine which dimension this face is perpendicular to
                    unsigned int perpDim = face / 2;

                    // Does not have BC if entity does extend in the perpendicular dimension
                    if (entityDir[perpDim]) {
                        // Entity extends in perpendicular dimension, so it doesn't reach this face
                        // Create NoBcFace to avoid null pointers
                        bcField[face] = std::make_shared<NoBcFace<Field_t>>(face);
                        continue;
                    }

                    // Apply the appropriate BC type for this face
                    switch (bcTypes[face]) {
                        case PERIODIC_FACE:
                            bcField[face] = std::make_shared<PeriodicFace<Field_t>>(face);
                            break;
                        case ZERO_FACE:
                            bcField[face] = std::make_shared<ZeroFace<Field_t>>(face);
                            break;
                        case CONSTANT_FACE:
                            // For CONSTANT_FACE, check if a value was provided
                            if (bcValues_m[face] == 0.0) {
                                throw std::runtime_error("CONSTANT_FACE BC on face " + std::to_string(face) +
                                    " requires a constant value. Use setFieldBC(BConds) to provide values.");
                            }
                            bcField[face] = std::make_shared<ConstantFace<Field_t>>(face, value_type(bcValues_m[face]));
                            break;
                        case EXTRAPOLATE_FACE:
                            // For EXTRAPOLATE_FACE, check if values were provided
                            if (bcValues_m[face] == 0.0 && bcSlopes_m[face] == 0.0) {
                                throw std::runtime_error("EXTRAPOLATE_FACE BC on face " + std::to_string(face) +
                                    " requires offset and slope values. Use setFieldBC(BConds) to provide values.");
                            }
                            bcField[face] = std::make_shared<ExtrapolateFace<Field_t>>(face, value_type(bcValues_m[face]), value_type(bcSlopes_m[face]));
                            break;
                        case NO_FACE:
                            // For NO_FACE, create NoBcFace
                            bcField[face] = std::make_shared<NoBcFace<Field_t>>(face);
                            break;
                        default:
                            // Create NoBcFace to avoid null pointers
                            bcField[face] = std::make_shared<NoBcFace<Field_t>>(face);
                            break;
                    }
                }

                // Debug: check if field is properly initialized
                auto& field = std::get<Is>(data_m);

                // Apply the boundary conditions to the field
                field.setFieldBC(bcField);
            }()), ...);
        }(std::make_index_sequence<NEntitys>{});
    }

        template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::setFieldBC(const std::array<FieldBC, 2*Dim>& bcTypes, std::array<T, 2*Dim> bcValues, std::array<T, 2*Dim> bcSlopes) {
        // Store the boundary condition types
        bcTypes_m = bcTypes;

        // Apply boundary conditions to each field in the container
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                // Get the entity type and its direction
                using EntityType = std::tuple_element_t<Is, EntityTypes>;
                constexpr std::array<bool, Dim> entityDir = EntityType{}.getDir();

                // Get the field type for this entity
                using Field_t = std::tuple_element_t<Is, FieldTuple>;
                using BConds_t = BConds<Field_t, Dim>;
                using value_type = typename Field_t::value_type;

                // Create boundary conditions for this field
                BConds_t bcField;

                // Domain boundary faces are numbered:
                // Face 0 = X lower (X=0 plane), Face 1 = X upper (X=max plane)
                // Face 2 = Y lower (Y=0 plane), Face 3 = Y upper (Y=max plane)
                // Face 4 = Z lower (Z=0 plane), Face 5 = Z upper (Z=max plane)
                //
                // An entity defined on a face F perpendicular to dimension d
                // can have a BC if the entity does NOT vary in dimension d (i.e., entityDir[d] == false)
                //
                // Examples:
                // - Vertex (entityDir = [false, false, false]): BCs on all faces
                // - EdgeX (entityDir = [true, false, false]): BCs on Y and Z faces (faces 2,3,4,5)
                // - FaceXY (entityDir = [true, true, false]): BCs on Z faces (faces 4,5)
                // - Hexahedron (entityDir = [true, true, true]): No BCs

                for (unsigned int face = 0; face < 2 * Dim; ++face) {
                    // Determine which dimension this face is perpendicular to
                    unsigned int perpDim = face / 2;

                    // Does not have BC if entity does extend in the perpendicular dimension
                    if (entityDir[perpDim]) {
                        // Entity extends in perpendicular dimension, so it doesn't reach this face
                        // Create NoBcFace to avoid null pointers
                        bcField[face] = std::make_shared<NoBcFace<Field_t>>(face);
                        continue;
                    }

                    // Apply the appropriate BC type for this face
                    switch (bcTypes[face]) {
                        case PERIODIC_FACE:
                            bcField[face] = std::make_shared<PeriodicFace<Field_t>>(face);
                            break;
                        case ZERO_FACE:
                            bcField[face] = std::make_shared<ZeroFace<Field_t>>(face);
                            break;
                        case CONSTANT_FACE:
                            // For CONSTANT_FACE, use the stored constant value
                            bcField[face] = std::make_shared<ConstantFace<Field_t>>(face, value_type(bcValues[face]));
                            break;
                        case EXTRAPOLATE_FACE:
                            // For EXTRAPOLATE_FACE, use stored offset and slope values
                            bcField[face] = std::make_shared<ExtrapolateFace<Field_t>>(face, value_type(bcValues[face]), value_type(bcSlopes[face]));
                            break;
                        case NO_FACE:
                            // For NO_FACE, create NoBcFace
                            bcField[face] = std::make_shared<NoBcFace<Field_t>>(face);
                            break;
                        default:
                            // Create NoBcFace to avoid null pointers
                            bcField[face] = std::make_shared<NoBcFace<Field_t>>(face);
                            break;
                    }
                }

                // Debug: check if field is properly initialized
                auto& field = std::get<Is>(data_m);

                // Apply the boundary conditions to the field
                field.setFieldBC(bcField);
            }()), ...);
        }(std::make_index_sequence<NEntitys>{});
    }

        // Set boundary conditions from BConds object
    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    template <typename Field>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::setFieldBC(const BConds<Field, Dim>& bconds) {
        // Extract BC types, offset values, and slope values from the BConds object that need them
        std::array<FieldBC, 2*Dim> bcTypes;
        std::array<T, 2*Dim> bcValues{};  // Store offset values for CONSTANT_FACE and EXTRAPOLATE_FACE BCs
        std::array<T, 2*Dim> bcSlopes{};  // Store slope values for EXTRAPOLATE_FACE BCs

        for (unsigned int face = 0; face < 2 * Dim; ++face) {
            auto bc = bconds[face];
            if (bc) {
                bcTypes[face] = bc->getBCType();

                // If it's a CONSTANT_FACE or EXTRAPOLATE_FACE, extract offset and slope values
                // Both inherit from ExtrapolateFace, so both have getOffset() and getSlope()
                if (bcTypes[face] == CONSTANT_FACE || bcTypes[face] == EXTRAPOLATE_FACE) {
                    auto* extrapolateFace = dynamic_cast<ExtrapolateFace<Field>*>(bc.get());
                    if (extrapolateFace) {
                        auto offset = extrapolateFace->getOffset();
                        auto slope = extrapolateFace->getSlope();

                        // Extract scalar from the field value type (might be DOFArray)
                        if constexpr (requires { offset[0]; }) {
                            bcValues[face] = offset[0];  // DOFArray case
                            bcSlopes[face] = slope[0];
                        } else {
                            bcValues[face] = offset;      // Scalar case
                            bcSlopes[face] = slope;
                        }
                    }
                }
            } else {
                bcTypes[face] = NO_FACE;
            }
        }

        // Now call the array-based setFieldBC with extracted types
        setFieldBC(bcTypes, bcValues, bcSlopes);
    }

    // Get boundary conditions as BConds object (for solver compatibility)
    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    BConds<Field<T, Dim, typename FEMContainer<T, Dim, EntityTypes, DOFNums>::Mesh_t, Cell>, Dim>
    FEMContainer<T, Dim, EntityTypes, DOFNums>::getFieldBC() const {
        using ScalarField_t = Field<T, Dim, Mesh_t, Cell>;
        BConds<ScalarField_t, Dim> bconds;

        // Construct BConds from stored types and values
        for (unsigned int face = 0; face < 2 * Dim; ++face) {
            switch (bcTypes_m[face]) {
                case PERIODIC_FACE:
                    bconds[face] = std::make_shared<PeriodicFace<ScalarField_t>>(face);
                    break;
                case ZERO_FACE:
                    bconds[face] = std::make_shared<ZeroFace<ScalarField_t>>(face);
                    break;
                case CONSTANT_FACE:
                    bconds[face] = std::make_shared<ConstantFace<ScalarField_t>>(face, bcValues_m[face]);
                    break;
                case EXTRAPOLATE_FACE:
                    bconds[face] = std::make_shared<ExtrapolateFace<ScalarField_t>>(face, bcValues_m[face], bcSlopes_m[face]);
                    break;
                case NO_FACE:
                default:
                    bconds[face] = std::make_shared<NoBcFace<ScalarField_t>>(face);
                    break;
            }
        }

        return bconds;
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::applyBC() {
        // Apply boundary conditions to each field
        std::apply([&](auto&... fields) {
            ((fields.getFieldBC().apply(fields)), ...);
        }, data_m);
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::assignGhostToPhysical() {
        // Assign ghost to physical for each field
        std::apply([&](auto&... fields) {
            ((fields.getFieldBC().assignGhostToPhysical(fields)), ...);
        }, data_m);
    }

    // Member function implementation for volume average
    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    T FEMContainer<T, Dim, EntityTypes, DOFNums>::getVolumeAverage() const {
        // Simple average using only the first field (typically vertices for P1 elements)
        // TODO: For proper FEM volume average with higher-order elements or mixed centering,
        //       this should be computed in LagrangeSpace using basis functions and quadrature.
        //       Currently this works for P1 Lagrange elements where only vertex DOFs exist.
        auto avg = std::get<0>(data_m).getVolumeAverage();
        // Extract the first DOF component (for P1 elements with 1 DOF per vertex, this is the only component)
        return avg[0];
    }


}