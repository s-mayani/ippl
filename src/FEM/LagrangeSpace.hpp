
namespace ippl {

    // Functors for LagrangeSpace parallel operations
    // Must be defined at namespace scope for CUDA compatibility

    // Functor for evaluateAx parallel_for loops
    template <typename T, typename DOFHandlerType, typename ViewA, typename ViewB,
              typename IndicesType, typename ElemIndicesView, typename MatrixType,
              typename BCTypesArray, std::size_t DofStartA, std::size_t DofEndA,
              std::size_t DofStartB, std::size_t DofEndB>
    struct LagrangeEvaluateAxFunctor {
        DOFHandlerType dofHandler;
        ElemIndicesView elemIndices;
        ViewA viewBC;
        ViewB view;
        ViewA resultView;
        MatrixType A_K;
        BCTypesArray bcTypes;
        int nghost;

        KOKKOS_INLINE_FUNCTION void operator()(const size_t index) const {
            using DOFMapping_t = typename DOFHandlerType::DOFMapping_t;

            // Get element index
            const size_t elementIndex = elemIndices(index);
            const IndicesType elementNDIndex =
                dofHandler.getLocalElementNDIndex(elementIndex, nghost);

            // Loop over local DOFs for entity type A and B
            for (size_t i = DofStartA; i < DofEndA; ++i) {
                // Get DOFMapping for local DOF i
                DOFMapping_t dofMap_i = dofHandler.getElementDOFMapping(i);

                // Handle boundary DOFs
                if (bcTypes[0] == CONSTANT_FACE && dofHandler.isDOFOnBoundary(elementIndex, i)) {
                    Kokkos::atomic_store(
                        &apply(resultView,
                               elementNDIndex + dofMap_i.entityLocalIndex)[dofMap_i.entityLocalDOF],
                        apply(viewBC,
                              elementNDIndex + dofMap_i.entityLocalIndex)[dofMap_i.entityLocalDOF]);
                    continue;
                } else if (bcTypes[0] == ZERO_FACE && dofHandler.isDOFOnBoundary(elementIndex, i)) {
                    continue;
                }

                for (size_t j = DofStartB; j < DofEndB; ++j) {
                    // Get DOFMapping for local DOF j
                    DOFMapping_t dofMap_j = dofHandler.getElementDOFMapping(j);

                    // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                    if (((bcTypes[0] == CONSTANT_FACE) || (bcTypes[0] == ZERO_FACE))
                        && dofHandler.isDOFOnBoundary(elementIndex, j)) {
                        continue;
                    }

                    T contrib =
                        A_K[i][j]
                        * apply(view, elementNDIndex
                                          + dofMap_j.entityLocalIndex)[dofMap_j.entityLocalDOF];
                    // Apply contribution to result
                    Kokkos::atomic_add(
                        &apply(resultView,
                               elementNDIndex + dofMap_i.entityLocalIndex)[dofMap_i.entityLocalDOF],
                        contrib);
                }
            }
        }
    };

    // Functors for LagrangeSpace parallel operations
    // Must be defined at namespace scope for CUDA compatibility

    // Functor for evaluateAx_lift parallel_for loops
    template <typename T, typename DOFHandlerType, typename ViewB, typename ViewA,
              typename IndicesType, typename ElemIndicesView, typename MatrixType,
              std::size_t DofStartA, std::size_t DofEndA, std::size_t DofStartB,
              std::size_t DofEndB>
    struct LagrangeEvaluateAx_liftFunctor {
        DOFHandlerType dofHandler;
        ElemIndicesView elemIndices;
        ViewB inputView_b;   // Input view for entity type B (used to read field values)
        ViewA resultView_a;  // Result view for entity type A (used to write results)
        MatrixType A_K;
        int nghost;

        KOKKOS_INLINE_FUNCTION void operator()(const size_t index) const {
            using DOFMapping_t = typename DOFHandlerType::DOFMapping_t;

            // Get element index
            const size_t elementIndex = elemIndices(index);
            const IndicesType elementNDIndex =
                dofHandler.getLocalElementNDIndex(elementIndex, nghost);

            // Loop over local DOFs for entity type A and B
            for (size_t i = DofStartA; i < DofEndA; ++i) {
                // Get DOFMapping for local DOF i
                DOFMapping_t dofMap_i = dofHandler.getElementDOFMapping(i);

                // Skip if on a row of the matrix
                if (dofHandler.isDOFOnBoundary(elementIndex, i)) {
                    continue;
                }

                for (size_t j = DofStartB; j < DofEndB; ++j) {
                    // Get DOFMapping for local DOF j
                    DOFMapping_t dofMap_j = dofHandler.getElementDOFMapping(j);

                    // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                    if (dofHandler.isDOFOnBoundary(elementIndex, j)) {
                        T contrib =
                            A_K[i][j]
                            * apply(inputView_b,
                                    elementNDIndex
                                        + dofMap_j.entityLocalIndex)[dofMap_j.entityLocalDOF];
                        // Apply contribution to result
                        Kokkos::atomic_add(
                            &apply(resultView_a,
                                   elementNDIndex
                                       + dofMap_i.entityLocalIndex)[dofMap_i.entityLocalDOF],
                            contrib);
                    }
                }
            }
        }
    };

    // Functor for evaluateLoadVector parallel_for loop
    template <typename T, typename DOFHandlerType, typename ViewB, typename ViewA,
              typename IndicesType, typename ElemIndicesView, typename BasisMatrixType,
              typename WeightsVectorType, typename BCTypesArray, std::size_t DofStartA,
              std::size_t DofEndA, std::size_t DofStartB, std::size_t DofEndB>
    struct LagrangeEvaluateLoadVectorFunctor {
        DOFHandlerType dofHandler;
        ElemIndicesView elemIndices;
        ViewB inputView_b;   // Input view for entity type B (used to read field values)
        ViewA resultView_a;  // Result view for entity type A (used to write results)
        BasisMatrixType basis_q;
        WeightsVectorType w;
        BCTypesArray bcTypes;
        T absDetDPhi;
        int nghost;

        KOKKOS_INLINE_FUNCTION void operator()(const size_t index) const {
            using DOFMapping_t = typename DOFHandlerType::DOFMapping_t;

            // Get element index
            const size_t elementIndex = elemIndices(index);

            const IndicesType elementNDIndex =
                dofHandler.getLocalElementNDIndex(elementIndex, nghost);

            // Loop over local DOFs for entity type A
            for (size_t i = DofStartA; i < DofEndA; ++i) {
                // Get DOFMapping for local DOF i
                DOFMapping_t dofMap_i = dofHandler.getElementDOFMapping(i);

                // Skip boundary DOFs (Zero and Constant Dirichlet BCs)
                if ((bcTypes[0] == CONSTANT_FACE || bcTypes[0] == ZERO_FACE)
                    && dofHandler.isDOFOnBoundary(elementIndex, i)) {
                    continue;
                }

                // calculate contribution of this elements DOF on enitityType A
                T contrib = 0;

                // Loop over local DOFs for entity type B
                for (size_t j = DofStartB; j < DofEndB; ++j) {
                    // Get DOFMapping for local DOF j
                    DOFMapping_t dofMap_j = dofHandler.getElementDOFMapping(j);

                    T val = 0;
                    for (size_t k = 0; k < BasisMatrixType::dim;
                         ++k) {  // QuadratureType::numElementNodes
                        // get field value at DOF of entity type B and interpolate to q_k
                        val += basis_q[k][j] * basis_q[k][i] * w[k];
                    }

                    // Apply contribution to result
                    contrib += absDetDPhi * val
                               * apply(inputView_b,
                                       elementNDIndex
                                           + dofMap_j.entityLocalIndex)[dofMap_j.entityLocalDOF];
                }

                // add the contribution of the element to the field atomically
                Kokkos::atomic_add(
                    &apply(resultView_a,
                           elementNDIndex + dofMap_i.entityLocalIndex)[dofMap_i.entityLocalDOF],
                    contrib);
            }
        }
    };

    // Functor for computeErrorL2 parallel_reduce loop
    template <typename T, unsigned Dim, typename DOFHandlerType, typename ViewA,
              typename ElementType, typename IndicesType, typename ElemIndicesView,
              typename BasisMatrixType, typename WeightsVectorType, typename QuadPointsType,
              typename SolutionFunctor, std::size_t DofStartA, std::size_t DofEndA>
    struct LagrangeComputeErrorL2Functor {
        DOFHandlerType dofHandler;
        ElemIndicesView elemIndices;
        ViewA view_a;
        BasisMatrixType basis_q;
        WeightsVectorType w;
        QuadPointsType q;
        SolutionFunctor u_sol;
        ElementType ref_element;
        T absDetDPhi;
        int nghost;
        Vector<T, Dim> hr_m;      // Mesh spacing
        Vector<T, Dim> origin_m;  // Mesh origin

        KOKKOS_INLINE_FUNCTION void operator()(const size_t index, T& local) const {
            using DOFMapping_t    = typename DOFHandlerType::DOFMapping_t;
            using vertex_points_t = typename ElementType::vertex_points_t;

            // Get element index
            const size_t elementIndex = elemIndices(index);
            const IndicesType localElementNDIndex =
                dofHandler.getLocalElementNDIndex(elementIndex, nghost);
            const IndicesType globalElementNDIndex = dofHandler.getElementNDIndex(elementIndex);

            // Compute element vertex points using DOFMapping and global element index
            // For Lagrange elements, the first DOFs are vertices
            vertex_points_t elementVertexPoints;
            constexpr size_t numVertices = ElementType::NumVertices;

            for (size_t v = 0; v < numVertices; ++v) {
                // Get the DOF mapping for this vertex
                DOFMapping_t vertexMapping = dofHandler.getElementDOFMapping(v);

                // Compute global vertex position
                for (size_t d = 0; d < Dim; ++d) {
                    size_t vertexGlobalIndex =
                        globalElementNDIndex[d] + vertexMapping.entityLocalIndex[d];
                    elementVertexPoints[v][d] = vertexGlobalIndex * hr_m[d] + origin_m[d];
                }
            }

            // contribution of this element to the error
            T contrib = 0;
            for (size_t k = 0; k < BasisMatrixType::dim; ++k) {  // QuadratureType::numElementNodes
                T val_u_sol = u_sol(ref_element.localToGlobal(elementVertexPoints, q[k]));

                T val_u_h = 0;
                for (size_t i = DofStartA; i < DofEndA; ++i) {
                    // Get DOFMapping for local DOF i
                    DOFMapping_t dofMap_i = dofHandler.getElementDOFMapping(i);

                    // get field value at DOF and interpolate to q_k
                    val_u_h +=
                        basis_q[k][i]
                        * apply(view_a, localElementNDIndex
                                            + dofMap_i.entityLocalIndex)[dofMap_i.entityLocalDOF];
                }

                contrib += w[k] * Kokkos::pow(val_u_sol - val_u_h, 2) * absDetDPhi;
            }
            local += contrib;
        }
    };

    // Functor for computeAvg parallel_reduce loop
    template <typename T, unsigned Dim, typename DOFHandlerType, typename ViewA,
              typename IndicesType, typename ElemIndicesView, typename BasisMatrixType,
              typename WeightsVectorType, std::size_t DofStartA, std::size_t DofEndA>
    struct LagrangeComputeAvgFunctor {
        DOFHandlerType dofHandler;
        ElemIndicesView elemIndices;
        ViewA view_a;
        BasisMatrixType basis_q;
        WeightsVectorType w;
        T absDetDPhi;
        int nghost;

        KOKKOS_INLINE_FUNCTION void operator()(const size_t index, T& local) const {
            using DOFMapping_t = typename DOFHandlerType::DOFMapping_t;

            // Get element index
            const size_t elementIndex = elemIndices(index);
            const IndicesType elementNDIndex =
                dofHandler.getLocalElementNDIndex(elementIndex, nghost);

            // contribution of this element to the average
            T contrib = 0;
            for (size_t k = 0; k < BasisMatrixType::dim; ++k) {  // QuadratureType::numElementNodes
                T val_u_h = 0;
                for (size_t i = DofStartA; i < DofEndA; ++i) {
                    // Get DOFMapping for local DOF i
                    DOFMapping_t dofMap_i = dofHandler.getElementDOFMapping(i);

                    // get field value at DOF and interpolate to q_k
                    val_u_h +=
                        basis_q[k][i]
                        * apply(view_a, elementNDIndex
                                            + dofMap_i.entityLocalIndex)[dofMap_i.entityLocalDOF];
                }

                contrib += w[k] * val_u_h * absDetDPhi;
            }
            local += contrib;
        }
    };

    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor,
    // and decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::LagrangeSpace(
        UniformCartesian<T, Dim>& mesh, ElementType& ref_element, const QuadratureType& quadrature,
        const Layout_t& layout)
        : FiniteElementSpace<T, Dim,
                             FiniteElementSpaceTraits<LagrangeSpaceTag, Dim, Order>::dofsPerElement,
                             ElementType, QuadratureType, FieldLHS, FieldRHS>(mesh, ref_element,
                                                                              quadrature)
        , dofHandler_m(mesh, layout) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");

        // Initialize the elementIndices view
        elementIndices = dofHandler_m.getElementIndices();
    }

    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::LagrangeSpace(
        UniformCartesian<T, Dim>& mesh, ElementType& ref_element, const QuadratureType& quadrature)
        : FiniteElementSpace<T, Dim,
                             FiniteElementSpaceTraits<LagrangeSpaceTag, Dim, Order>::dofsPerElement,
                             ElementType, QuadratureType, FieldLHS, FieldRHS>(mesh, ref_element,
                                                                              quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");
    }

    // LagrangeSpace initializer, to be made available to the FEMPoissonSolver
    // such that we can call it from setRhs.
    // Sets the correct mesh ad decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::initialize(
        UniformCartesian<T, Dim>& mesh, const Layout_t& layout) {
        FiniteElementSpace<T, Dim,
                           FiniteElementSpaceTraits<LagrangeSpaceTag, Dim, Order>::dofsPerElement,
                           ElementType, QuadratureType, FieldLHS, FieldRHS>::setMesh(mesh);

        // Initialize the DOFHandler
        dofHandler_m.initialize(mesh, layout);

        // Initialize the elementIndices view
        elementIndices = dofHandler_m.getElementIndices();
    }

    ///////////////////////////////////////////////////////////////////////
    /// Degree of Freedom operations //////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                         FieldRHS>::numGlobalDOFs() const {
        size_t num_global_dofs = 1;
        for (size_t d = 0; d < Dim; ++d) {
            num_global_dofs *= this->nr_m[d] * Order;
        }

        return num_global_dofs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getLocalDOFIndex(
        const size_t& elementIndex, const size_t& globalDOFIndex) const {
        static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");
        // TODO fix not order independent, only works for order 1
        // static_assert(Order == 1, "Only order 1 is supported at the moment");

        // Get all the global DOFs for the element
        const Vector<size_t, numElementDOFs> global_dofs =
            this->LagrangeSpace::getGlobalDOFIndices(elementIndex);

        // Find the global DOF in the vector and return the local DOF index
        // Note: It is important that this only works because the global_dofs
        // are already arranged in the correct order from getGlobalDOFIndices
        for (size_t i = 0; i < global_dofs.dim; ++i) {
            if (global_dofs[i] == globalDOFIndex) {
                return i;
            }
        }
        // commented this due to this being on device
        // however, it would be good to throw an error in this case
        // throw IpplException("LagrangeSpace::getLocalDOFIndex()",
        //                    "FEM Lagrange Space: Global DOF not found in specified element");
        return 0;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::getGlobalDOFIndex(const size_t& elementIndex,
                                               const size_t& localDOFIndex) const {
        const auto global_dofs = this->LagrangeSpace::getGlobalDOFIndices(elementIndex);

        return global_dofs[localDOFIndex];
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION Vector<size_t, LagrangeSpace<T, Dim, Order, ElementType, QuadratureType,
                                                 FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::getLocalDOFIndices() const {
        Vector<size_t, numElementDOFs> localDOFs;

        for (size_t dof = 0; dof < numElementDOFs; ++dof) {
            localDOFs[dof] = dof;
        }

        return localDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION Vector<size_t, LagrangeSpace<T, Dim, Order, ElementType, QuadratureType,
                                                 FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::getGlobalDOFIndices(const size_t& elementIndex) const {
        Vector<size_t, numElementDOFs> globalDOFs(0);

        // get element pos
        indices_t elementPos = this->getElementNDIndex(elementIndex);

        // Compute the vector to multiply the ndindex with
        ippl::Vector<size_t, Dim> vec(1);
        for (size_t d = 1; d < dim; ++d) {
            for (size_t d2 = d; d2 < Dim; ++d2) {
                vec[d2] *= this->nr_m[d - 1];
            }
        }
        vec *= Order;  // Multiply each dimension by the order
        size_t smallestGlobalDOF = elementPos.dot(vec);

        // Add the vertex DOFs
        globalDOFs[0] = smallestGlobalDOF;
        globalDOFs[1] = smallestGlobalDOF + Order;

        if constexpr (Dim >= 2) {
            globalDOFs[2] = globalDOFs[1] + this->nr_m[0] * Order;
            globalDOFs[3] = globalDOFs[0] + this->nr_m[0] * Order;
        }
        if constexpr (Dim >= 3) {
            globalDOFs[4] = globalDOFs[0] + this->nr_m[1] * this->nr_m[0] * Order;
            globalDOFs[5] = globalDOFs[1] + this->nr_m[1] * this->nr_m[0] * Order;
            globalDOFs[6] = globalDOFs[2] + this->nr_m[1] * this->nr_m[0] * Order;
            globalDOFs[7] = globalDOFs[3] + this->nr_m[1] * this->nr_m[0] * Order;
        }

        if constexpr (Order > 1) {
            // If the order is greater than 1, there are edge and face DOFs, otherwise the work is
            // done

            // Add the edge DOFs
            if constexpr (Dim >= 2) {
                for (size_t i = 0; i < Order - 1; ++i) {
                    globalDOFs[8 + i]                   = globalDOFs[0] + i + 1;
                    globalDOFs[8 + Order - 1 + i]       = globalDOFs[1] + (i + 1) * this->nr_m[1];
                    globalDOFs[8 + 2 * (Order - 1) + i] = globalDOFs[2] - (i + 1);
                    globalDOFs[8 + 3 * (Order - 1) + i] = globalDOFs[3] - (i + 1) * this->nr_m[1];
                }
            }
            if constexpr (Dim >= 3) {
                // TODO
            }

            // Add the face DOFs
            if constexpr (Dim >= 2) {
                for (size_t i = 0; i < Order - 1; ++i) {
                    for (size_t j = 0; j < Order - 1; ++j) {
                        // TODO CHECK
                        globalDOFs[8 + 4 * (Order - 1) + i * (Order - 1) + j] =
                            globalDOFs[0] + (i + 1) + (j + 1) * this->nr_m[1];
                        globalDOFs[8 + 4 * (Order - 1) + (Order - 1) * (Order - 1) + i * (Order - 1)
                                   + j] = globalDOFs[1] + (i + 1) + (j + 1) * this->nr_m[1];
                        globalDOFs[8 + 4 * (Order - 1) + 2 * (Order - 1) * (Order - 1)
                                   + i * (Order - 1) + j] =
                            globalDOFs[2] - (i + 1) + (j + 1) * this->nr_m[1];
                        globalDOFs[8 + 4 * (Order - 1) + 3 * (Order - 1) * (Order - 1)
                                   + i * (Order - 1) + j] =
                            globalDOFs[3] - (i + 1) + (j + 1) * this->nr_m[1];
                    }
                }
            }
        }

        return globalDOFs;
    }

    ///////////////////////////////////////////////////////////////////////
    /// Basis functions and gradients /////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                           FieldRHS>::point_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::getRefElementDOFLocation(const size_t& localDOF) const {
        // Assert that the local DOF index is valid
        assert(localDOF < numElementDOFs && "The local DOF index is invalid");

        // Use precomputed DOF locations
        return dofLocations_m[localDOF];
    }

    // TODO make function branchless for performance
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION T
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        evaluateRefElementShapeFunction(
            const size_t& localDOF,
            const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                FieldRHS>::point_t& localPoint) const {
        // Assert that the local DOF index is valid.
        assert(localDOF < numElementDOFs && "The local DOF index is invalid");

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the DOF location on the reference element
        const point_t ref_element_point = getRefElementDOFLocation(localDOF);

        // For Lagrange elements, the shape function is a tensor product of 1D Lagrange polynomials
        T product = 1;

        for (size_t d = 0; d < Dim; d++) {
            // Compute 1D Lagrange basis polynomial in dimension d
            T basis_1d = 1.0;

            // Loop over all nodes in this dimension to construct Lagrange polynomial
            for (unsigned k = 0; k <= Order; ++k) {
                T node_k = static_cast<T>(k) / static_cast<T>(Order);

                // Skip if this is the node corresponding to ref_element_point[d]
                if (Kokkos::abs(ref_element_point[d] - node_k) < 1e-10) {
                    continue;
                }

                // Lagrange basis: product of (x - x_k) / (x_i - x_k)
                basis_1d *= (localPoint[d] - node_k) / (ref_element_point[d] - node_k);
            }

            product *= basis_1d;
        }

        return product;
    }

    // TODO make function branchless for performance
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                           FieldRHS>::point_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        evaluateRefElementShapeFunctionGradient(
            const size_t& localDOF,
            const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                FieldRHS>::point_t& localPoint) const {
        // Assert that the local DOF index is valid.
        assert(localDOF < numElementDOFs && "The local DOF index is invalid");

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the DOF location on the reference element
        const point_t ref_element_point = getRefElementDOFLocation(localDOF);

        point_t gradient;

        // For Lagrange elements, gradient is tensor product with one derivative
        // grad_i(phi) = [phi_1, phi_2, ..., dphi_i/dx_i, ..., phi_Dim]
        for (size_t d = 0; d < Dim; d++) {
            T product = 1;

            for (size_t d2 = 0; d2 < Dim; d2++) {
                if (d2 == d) {
                    // Compute derivative of 1D Lagrange basis in this dimension
                    T basis_deriv_1d = 0.0;

                    // Loop over all nodes to construct derivative
                    for (unsigned k = 0; k <= Order; ++k) {
                        T node_k = static_cast<T>(k) / static_cast<T>(Order);

                        // Skip if this is the node corresponding to ref_element_point[d]
                        if (Kokkos::abs(ref_element_point[d] - node_k) < 1e-10) {
                            continue;
                        }

                        // Compute derivative using product rule
                        T term = 1.0 / (ref_element_point[d] - node_k);

                        for (unsigned j = 0; j <= Order; ++j) {
                            T node_j = static_cast<T>(j) / static_cast<T>(Order);

                            if (Kokkos::abs(ref_element_point[d] - node_j) < 1e-10 || j == k) {
                                continue;
                            }

                            term *= (localPoint[d] - node_j) / (ref_element_point[d] - node_j);
                        }

                        basis_deriv_1d += term;
                    }

                    product *= basis_deriv_1d;
                } else {
                    // Compute 1D Lagrange basis in other dimensions
                    T basis_1d = 1.0;

                    for (unsigned k = 0; k <= Order; ++k) {
                        T node_k = static_cast<T>(k) / static_cast<T>(Order);

                        if (Kokkos::abs(ref_element_point[d2] - node_k) < 1e-10) {
                            continue;
                        }

                        basis_1d *= (localPoint[d2] - node_k) / (ref_element_point[d2] - node_k);
                    }

                    product *= basis_1d;
                }
            }

            gradient[d] = product;
        }

        return gradient;
    }

    ///////////////////////////////////////////////////////////////////////
    /// Assembly operations ///////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    // TODO Fix boundary conditions for result field (not set correctly before apply)

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)

        // TODO check if we need to set BCs the same as field
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get boundary conditions from field
        // Copy the array by value so it can be safely used in device code
        const std::array<FieldBC, 2 * Dim> bcTypes = field.getFieldBCTypes();

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Copy member variables for capture in kernel
        auto dofHandler  = dofHandler_m;
        auto elemIndices = elementIndices;

        // Helper lambda to process a single entity type pair
        auto processEntityPair = [&]<typename EntityTypeA, typename EntityTypeB>() {
            // Get DOF ranges
            constexpr size_t dofStart_a = DOFHandler_t::template getEntityDOFStart<EntityTypeA>();
            constexpr size_t dofEnd_a   = DOFHandler_t::template getEntityDOFEnd<EntityTypeA>();
            constexpr size_t dofStart_b = DOFHandler_t::template getEntityDOFStart<EntityTypeB>();
            constexpr size_t dofEnd_b   = DOFHandler_t::template getEntityDOFEnd<EntityTypeB>();

            // Get views for these entity types
            using ViewType_a = std::remove_cv_t<
                std::remove_reference_t<decltype(field.template getView<EntityTypeA>())>>;
            using ViewType_b = std::remove_cv_t<
                std::remove_reference_t<decltype(field.template getView<EntityTypeB>())>>;

            ViewType_a resultView = resultField.template getView<EntityTypeA>();

            ViewType_a viewBC =
                field.template getView<EntityTypeA>();  // Needed for constant Dirichlet BCs
            ViewType_b view = field.template getView<EntityTypeB>();

            // Get execution space and policy type
            using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            // Create functor type
            using functor_t = LagrangeEvaluateAxFunctor<
                T, decltype(dofHandler), ViewType_a, ViewType_b, indices_t, decltype(elemIndices),
                decltype(A_K), decltype(bcTypes), dofStart_a, dofEnd_a, dofStart_b, dofEnd_b>;

            Kokkos::parallel_for(
                "Loop over elements", policy_type(0, elemIndices.extent(0)),
                functor_t{dofHandler, elemIndices, viewBC, view, resultView, A_K, bcTypes, nghost});
        };

        // Iterate over all entity type pairs using compile-time double loop
        constexpr size_t numTypes = DOFHandler_t::numEntityTypes;

        // Execute the double loop over all entity type pairs
        [&]<size_t... IJs>(std::index_sequence<IJs...>) {
            (
                [&] {
                    constexpr size_t I = IJs / numTypes;
                    constexpr size_t J = IJs % numTypes;
                    using EntityTypeA = std::tuple_element_t<I, typename DOFHandler_t::EntityTypes>;
                    using EntityTypeB = std::tuple_element_t<J, typename DOFHandler_t::EntityTypes>;
                    processEntityPair.template operator()<EntityTypeA, EntityTypeB>();
                }(),
                ...);
        }(std::make_index_sequence<numTypes * numTypes>{});

        IpplTimings::stopTimer(outer_loop);

        // TODO add Periodic BC handling for entity type views
        if (bcTypes[0] == PERIODIC_FACE) {
            resultField.accumulateHalo();
            resultField.applyBC();
            resultField.assignGhostToPhysical();
        } else {
            resultField.accumulateHalo_noghost();
        }

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_lower(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType                 = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->LagrangeSpace::getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) = apply(view, I_nd);
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        if (global_dofs[i] >= global_dofs[j]) {
                            continue;
                        }

                        // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                        if (((bcType == ZERO_FACE) || (bcType == CONSTANT_FACE))
                            && this->isDOFOnBoundary(J_nd)) {
                            continue;
                        }

                        // get the appropriate index for the Kokkos view of the field
                        for (unsigned d = 0; d < Dim; ++d) {
                            J_nd[d] = J_nd[d] - ldom[d].first() + nghost;
                        }

                        apply(resultView, I_nd) += A_K[i][j] * apply(view, J_nd);
                    }
                }
            });
        IpplTimings::stopTimer(outer_loop);

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_upper(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType                 = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->LagrangeSpace::getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) = apply(view, I_nd);
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        if (global_dofs[i] <= global_dofs[j]) {
                            continue;
                        }

                        // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                        if (((bcType == ZERO_FACE) || (bcType == CONSTANT_FACE))
                            && this->isDOFOnBoundary(J_nd)) {
                            continue;
                        }

                        // get the appropriate index for the Kokkos view of the field
                        for (unsigned d = 0; d < Dim; ++d) {
                            J_nd[d] = J_nd[d] - ldom[d].first() + nghost;
                        }

                        apply(resultView, I_nd) += A_K[i][j] * apply(view, J_nd);
                    }
                }
            });
        IpplTimings::stopTimer(outer_loop);

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_upperlower(FieldLHS& field,
                                                            F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType                 = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->LagrangeSpace::getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) = apply(view, I_nd);
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < i; ++j) {
                        J_nd = global_dof_ndindices[j];

                        // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                        if (((bcType == ZERO_FACE) || (bcType == CONSTANT_FACE))
                            && this->isDOFOnBoundary(J_nd)) {
                            continue;
                        }

                        // get the appropriate index for the Kokkos view of the field
                        for (unsigned d = 0; d < Dim; ++d) {
                            J_nd[d] = J_nd[d] - ldom[d].first() + nghost;
                        }

                        apply(resultView, I_nd) += A_K[i][j] * apply(view, J_nd);
                        apply(resultView, J_nd) += A_K[j][i] * apply(view, I_nd);
                    }
                }
            });
        IpplTimings::stopTimer(outer_loop);

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_inversediag(FieldLHS& field,
                                                             F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<T, numElementDOFs> A_K_diag;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            A_K_diag[i] = 0.0;
            for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                A_K_diag[i] += w[k] * evalFunction(i, i, grad_b_q[k]);
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType                 = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->LagrangeSpace::getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) = 1.0;
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }
                    // sum up all contributions of element matrix
                    apply(resultView, I_nd) += A_K_diag[i];
                }
            });

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        // apply the inverse diagonal after already summed all contributions from element matrices
        using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
        ippl::parallel_for(
            "Loop over result view to apply inverse", field.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const index_array_type& args) {
                if (apply(resultView, args) != 0.0) {
                    apply(resultView, args) = (1.0 / apply(resultView, args)) * apply(view, args);
                }
            });
        IpplTimings::stopTimer(outer_loop);

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_diag(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<T, numElementDOFs> A_K_diag;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            A_K_diag[i] = 0.0;
            for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                A_K_diag[i] += w[k] * evalFunction(i, i, grad_b_q[k]);
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType                 = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->LagrangeSpace::getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) = apply(view, I_nd);
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }
                    apply(resultView, I_nd) += A_K_diag[i] * apply(view, I_nd);
                }
            });
        IpplTimings::stopTimer(outer_loop);

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_lift(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx_lift = IpplTimings::getTimer("evaluateAx_lift");
        IpplTimings::startTimer(evalAx_lift);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // start a timer
        static IpplTimings::TimerRef outer_loop =
            IpplTimings::getTimer("evaluateAx_lift: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Copy member variables for capture in kernel
        auto dofHandler  = dofHandler_m;
        auto elemIndices = elementIndices;

        // Helper lambda to process a single entity type pair
        auto processEntityPair = [&]<typename EntityTypeA, typename EntityTypeB>() {
            // Get DOF ranges
            constexpr size_t dofStart_a = DOFHandler_t::template getEntityDOFStart<EntityTypeA>();
            constexpr size_t dofEnd_a   = DOFHandler_t::template getEntityDOFEnd<EntityTypeA>();
            constexpr size_t dofStart_b = DOFHandler_t::template getEntityDOFStart<EntityTypeB>();
            constexpr size_t dofEnd_b   = DOFHandler_t::template getEntityDOFEnd<EntityTypeB>();

            // Get views for these entity types
            using ViewType_a = std::remove_cv_t<
                std::remove_reference_t<decltype(field.template getView<EntityTypeA>())>>;
            using ViewType_b = std::remove_cv_t<
                std::remove_reference_t<decltype(field.template getView<EntityTypeB>())>>;

            ViewType_a resultView_a = resultField.template getView<EntityTypeA>();
            ViewType_b view_b       = field.template getView<EntityTypeB>();

            // Get execution space and policy type
            using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            // Create functor type
            using functor_t =
                LagrangeEvaluateAx_liftFunctor<T, decltype(dofHandler), ViewType_b, ViewType_a,
                                               indices_t, decltype(elemIndices), decltype(A_K),
                                               dofStart_a, dofEnd_a, dofStart_b, dofEnd_b>;

            Kokkos::parallel_for(
                "Loop over elements", policy_type(0, elemIndices.extent(0)),
                functor_t{dofHandler, elemIndices, view_b, resultView_a, A_K, nghost});
        };

        // Iterate over all entity type pairs using compile-time double loop
        constexpr size_t numTypes = DOFHandler_t::numEntityTypes;

        // Execute the double loop over all entity type pairs
        [&]<size_t... IJs>(std::index_sequence<IJs...>) {
            (
                [&] {
                    constexpr size_t I = IJs / numTypes;
                    constexpr size_t J = IJs % numTypes;
                    using EntityTypeA = std::tuple_element_t<I, typename DOFHandler_t::EntityTypes>;
                    using EntityTypeB = std::tuple_element_t<J, typename DOFHandler_t::EntityTypes>;
                    processEntityPair.template operator()<EntityTypeA, EntityTypeB>();
                }(),
                ...);
        }(std::make_index_sequence<numTypes * numTypes>{});

        IpplTimings::stopTimer(outer_loop);

        // Accumulate halo regions
        resultField.accumulateHalo();

        IpplTimings::stopTimer(evalAx_lift);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                       FieldRHS>::evaluateLoadVector(FieldRHS& field) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalLoadV = IpplTimings::getTimer("evaluateLoadVector");
        IpplTimings::startTimer(evalLoadV);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Get domain information and ghost cells
        const int nghost = field.getNghost();

        // Get boundary conditions from field
        // Copy the array by value so it can be safely used in device code
        const std::array<FieldBC, 2 * Dim> bcTypes = field.getFieldBCTypes();

        // Create temp_field using the same mesh and layout as the input field
        // to ensure compatible MPI communicators
        FieldRHS temp_field(field.get_mesh(), field.getLayout(), nghost);
        temp_field = 0.0;  // Initialize to zero before atomic adds

        temp_field.setFieldBC(field.getFieldBC());

        // start a timer
        static IpplTimings::TimerRef outer_loop =
            IpplTimings::getTimer("evaluateLoadVector: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Copy member variables for capture in kernel
        auto dofHandler  = dofHandler_m;
        auto elemIndices = elementIndices;

        // Helper lambda to process a single entity type pair
        auto processEntityPair = [&]<typename EntityTypeA, typename EntityTypeB>() {
            // Get DOF ranges
            constexpr size_t dofStart_a = DOFHandler_t::template getEntityDOFStart<EntityTypeA>();
            constexpr size_t dofEnd_a   = DOFHandler_t::template getEntityDOFEnd<EntityTypeA>();
            constexpr size_t dofStart_b = DOFHandler_t::template getEntityDOFStart<EntityTypeB>();
            constexpr size_t dofEnd_b   = DOFHandler_t::template getEntityDOFEnd<EntityTypeB>();

            // Get views for these entity types
            using ViewType_a = std::remove_cv_t<
                std::remove_reference_t<decltype(field.template getView<EntityTypeA>())>>;
            using ViewType_b = std::remove_cv_t<
                std::remove_reference_t<decltype(field.template getView<EntityTypeB>())>>;

            ViewType_b view_b       = field.template getView<EntityTypeB>();
            ViewType_a resultView_a = temp_field.template getView<EntityTypeA>();

            // Get execution space and policy type
            using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            // Create functor type
            using functor_t =
                LagrangeEvaluateLoadVectorFunctor<T, decltype(dofHandler), ViewType_b, ViewType_a,
                                                  indices_t, decltype(elemIndices),
                                                  decltype(basis_q), decltype(w), decltype(bcTypes),
                                                  dofStart_a, dofEnd_a, dofStart_b, dofEnd_b>;

            Kokkos::parallel_for("Loop over elements", policy_type(0, elemIndices.extent(0)),
                                 functor_t{dofHandler, elemIndices, view_b, resultView_a, basis_q,
                                           w, bcTypes, absDetDPhi, nghost});

            // Fence to ensure kernel completion before proceeding
            Kokkos::fence();
        };

        // Iterate over all entity type pairs using compile-time double loop
        constexpr size_t numTypes = DOFHandler_t::numEntityTypes;

        // Execute the double loop over all entity type pairs
        [&]<size_t... IJs>(std::index_sequence<IJs...>) {
            (
                [&] {
                    constexpr size_t I = IJs / numTypes;
                    constexpr size_t J = IJs % numTypes;
                    using EntityTypeA = std::tuple_element_t<I, typename DOFHandler_t::EntityTypes>;
                    using EntityTypeB = std::tuple_element_t<J, typename DOFHandler_t::EntityTypes>;

                    processEntityPair.template operator()<EntityTypeA, EntityTypeB>();
                }(),
                ...);
        }(std::make_index_sequence<numTypes * numTypes>{});

        IpplTimings::stopTimer(outer_loop);

        // temp_field.accumulateHalo();

        // TODO add Periodic BC handling for entity type views
        if (bcTypes[0] == PERIODIC_FACE) {
            temp_field.accumulateHalo();
            temp_field.applyBC();
        }

        field = temp_field;

        IpplTimings::stopTimer(evalLoadV);
    }

    ///////////////////////////////////////////////////////////////////////
    /// Functions for error computations, etc. ////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    T LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::computeErrorL2(
        const FieldLHS& u_h, const F& u_sol) const {
        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException(
                "LagrangeSpace::computeErrorL2()",
                "Order of quadrature rule for error computation should be > 2*p + 1");
        }

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Variable to sum the error to
        T error = 0;

        // Get ghost cells
        const int nghost = u_h.getNghost();

        // Non-parallel version for higher-order elements
        // Create mirror views once before the element loop
        auto createMirrorViews = [&]<typename EntityType>() {
            auto view      = u_h.template getView<EntityType>();
            auto view_host = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(view_host, view);
            return view_host;
        };

        // Create tuple of mirror views for all entity types
        constexpr size_t numTypes = DOFHandler_t::numEntityTypes;
        auto mirror_views_tuple   = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(
                createMirrorViews.template
                operator()<std::tuple_element_t<Is, typename DOFHandler_t::EntityTypes>>()...);
        }(std::make_index_sequence<numTypes>{});

        // Loop over all elements on the host
        size_t numElements = elementIndices.extent(0);

        for (size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
            using DOFMapping_t    = typename DOFHandler_t::DOFMapping_t;
            using vertex_points_t = typename ElementType::vertex_points_t;

            // Get element index
            const size_t elementIndex = elementIndices(elemIdx);
            const indices_t localElementNDIndex =
                dofHandler_m.getLocalElementNDIndex(elementIndex, nghost);
            const indices_t globalElementNDIndex = dofHandler_m.getElementNDIndex(elementIndex);

            // Compute element vertex points
            vertex_points_t elementVertexPoints;
            constexpr size_t numVertices = ElementType::NumVertices;

            for (size_t v = 0; v < numVertices; ++v) {
                DOFMapping_t vertexMapping = dofHandler_m.getElementDOFMapping(v);
                for (size_t d = 0; d < Dim; ++d) {
                    size_t vertexGlobalIndex =
                        globalElementNDIndex[d] + vertexMapping.entityLocalIndex[d];
                    elementVertexPoints[v][d] =
                        vertexGlobalIndex * this->hr_m[d] + this->origin_m[d];
                }
            }

            // Collect all DOF values for this element
            Vector<T, numElementDOFs> element_dof_values(0.0);

            // Helper lambda to collect DOF values from a specific entity type
            auto collectDOFsFromEntityType = [&]<size_t EntityIdx>() {
                using EntityType =
                    std::tuple_element_t<EntityIdx, typename DOFHandler_t::EntityTypes>;
                constexpr size_t dofStart = DOFHandler_t::template getEntityDOFStart<EntityType>();
                constexpr size_t dofEnd   = DOFHandler_t::template getEntityDOFEnd<EntityType>();

                auto& view_host = std::get<EntityIdx>(mirror_views_tuple);

                for (size_t i = dofStart; i < dofEnd; ++i) {
                    DOFMapping_t dofMap_i = dofHandler_m.getElementDOFMapping(i);
                    element_dof_values[i] =
                        apply(view_host, localElementNDIndex
                                             + dofMap_i.entityLocalIndex)[dofMap_i.entityLocalDOF];
                }
            };

            // Loop over all entity types to collect DOF values
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (collectDOFsFromEntityType.template operator()<Is>(), ...);
            }(std::make_index_sequence<numTypes>{});

            // Compute error contribution from this element
            T element_error = 0;
            for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                // Evaluate exact solution at quadrature point
                T val_u_sol = u_sol(this->ref_element_m.localToGlobal(elementVertexPoints, q[k]));

                // Evaluate numerical solution at quadrature point using all DOFs
                T val_u_h = 0;
                for (size_t i = 0; i < numElementDOFs; ++i) {
                    val_u_h += basis_q[k][i] * element_dof_values[i];
                }

                element_error += w[k] * Kokkos::pow(val_u_sol - val_u_h, 2) * absDetDPhi;
            }

            error += element_error;
        }

        // MPI reduce
        T global_error = 0.0;
        Comm->allreduce(error, global_error, 1, std::plus<T>());

        return Kokkos::sqrt(global_error);
        /*
        Old parallel version

        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException(
                "LagrangeSpace::computeErrorL2()",
                "Order of quadrature rule for error computation should be > 2*p + 1");
        }

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Variable to sum the error to
        T error = 0;

        // Get ghost cells
        const int nghost = u_h.getNghost();

        // Copy member variables for capture in kernel
        auto dofHandler = dofHandler_m;
        auto elemIndices = elementIndices;
        auto ref_element = this->ref_element_m;
        auto hr_m = this->hr_m;
        auto origin_m = this->origin_m;

        // Helper lambda to process a single entity type
        auto processEntityType = [&]<typename EntityTypeA>() {
            // Get DOF ranges
            constexpr size_t dofStart_a = DOFHandler_t::template getEntityDOFStart<EntityTypeA>();
            constexpr size_t dofEnd_a   = DOFHandler_t::template getEntityDOFEnd<EntityTypeA>();

            // Get view for this entity type
            using ViewType_a = std::remove_cv_t<std::remove_reference_t<decltype(u_h.template
        getView<EntityTypeA>())>>; ViewType_a view_a = u_h.template getView<EntityTypeA>();

            // Get execution space and policy type
            using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            // Create functor type
            using functor_t = LagrangeComputeErrorL2Functor<
                T, Dim, decltype(dofHandler), ViewType_a, decltype(ref_element),
                indices_t, decltype(elemIndices), decltype(basis_q), decltype(w), decltype(q), F,
                dofStart_a, dofEnd_a>;

            T local_error = 0;
            Kokkos::parallel_reduce(
                "Compute error over elements", policy_type(0, elemIndices.extent(0)),
                functor_t{dofHandler, elemIndices, view_a, basis_q, w, q, u_sol, ref_element,
        absDetDPhi, nghost, hr_m, origin_m}, Kokkos::Sum<T>(local_error));

            error += local_error;
        };

        // Iterate over all entity types using compile-time loop
        constexpr size_t numTypes = DOFHandler_t::numEntityTypes;

        // Execute the loop over all entity types
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ([&] {
                using EntityTypeA = std::tuple_element_t<Is, typename DOFHandler_t::EntityTypes>;
                processEntityType.template operator()<EntityTypeA>();
            }(), ...);
        }(std::make_index_sequence<numTypes>{});

        // MPI reduce
        T global_error = 0.0;
        Comm->allreduce(error, global_error, 1, std::plus<T>());

        return Kokkos::sqrt(global_error);

        */
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    T LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::computeAvg(
        const FieldLHS& u_h) const {
        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException(
                "LagrangeSpace::computeAvg()",
                "Order of quadrature rule for error computation should be > 2*p + 1");
        }

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Variable to sum the average to
        T avg = 0;

        // Get ghost cells
        const int nghost = u_h.getNghost();

        // Copy member variables for capture in kernel
        auto dofHandler  = dofHandler_m;
        auto elemIndices = elementIndices;

        // Helper lambda to process a single entity type
        auto processEntityType = [&]<typename EntityTypeA>() {
            // Get DOF ranges
            constexpr size_t dofStart_a = DOFHandler_t::template getEntityDOFStart<EntityTypeA>();
            constexpr size_t dofEnd_a   = DOFHandler_t::template getEntityDOFEnd<EntityTypeA>();

            // Get view for this entity type
            using ViewType_a = std::remove_cv_t<
                std::remove_reference_t<decltype(u_h.template getView<EntityTypeA>())>>;
            ViewType_a view_a = u_h.template getView<EntityTypeA>();

            // Get execution space and policy type
            using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            // Create functor type
            using functor_t =
                LagrangeComputeAvgFunctor<T, Dim, decltype(dofHandler), ViewType_a, indices_t,
                                          decltype(elemIndices), decltype(basis_q), decltype(w),
                                          dofStart_a, dofEnd_a>;

            T local_avg = 0;
            Kokkos::parallel_reduce(
                "Compute average over elements", policy_type(0, elemIndices.extent(0)),
                functor_t{dofHandler, elemIndices, view_a, basis_q, w, absDetDPhi, nghost},
                Kokkos::Sum<T>(local_avg));

            avg += local_avg;
        };

        // Iterate over all entity types using compile-time loop
        constexpr size_t numTypes = DOFHandler_t::numEntityTypes;

        // Execute the loop over all entity types
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (
                [&] {
                    using EntityTypeA =
                        std::tuple_element_t<Is, typename DOFHandler_t::EntityTypes>;
                    processEntityType.template operator()<EntityTypeA>();
                }(),
                ...);
        }(std::make_index_sequence<numTypes>{});

        // MPI reduce
        T global_avg = 0.0;
        Comm->allreduce(avg, global_avg, 1, std::plus<T>());

        return global_avg;
    }

    ///////////////////////////////////////////////////////////////////////
    /// Device struct definitions /////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    // Function to return the device struct of this Lagrange Space
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::DeviceStruct
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getDeviceMirror()
        const {
        DeviceStruct space_mirror;
        space_mirror.nr_m           = this->nr_m;
        space_mirror.ref_element_m  = this->ref_element_m;
        space_mirror.dofLocations_m = this->dofLocations_m;
        return space_mirror;
    }

    // I don't know how to avoid code duplication here...
    // Make sure that any changes in getLocalDOFIndex, getGlobalDOFIndices,
    // evaluateRefElementShapeFunction, and getMeshVertexNDIndex from the
    // parent class FiniteElementSpace get propagated here.

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::DeviceStruct::getLocalDOFIndex(const indices_t& elementNDIndex,
                                                            const size_t& globalDOFIndex) const {
        static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");
        // TODO fix not order independent, only works for order 1
        // static_assert(Order == 1, "Only order 1 is supported at the moment");

        // Get all the global DOFs for the element
        const Vector<size_t, numElementDOFs> global_dofs =
            this->getGlobalDOFIndices(elementNDIndex);

        // Find the global DOF in the vector and return the local DOF index
        // Note: It is important that this only works because the global_dofs
        // are already arranged in the correct order from getGlobalDOFIndices
        for (size_t i = 0; i < global_dofs.dim; ++i) {
            if (global_dofs[i] == globalDOFIndex) {
                return i;
            }
        }
        // commented this due to this being on device
        // however, it would be good to throw an error in this case
        // throw IpplException("LagrangeSpace::getLocalDOFIndex()",
        //                    "FEM Lagrange Space: Global DOF not found in specified element");
        // FIXME: Take a look at Kokkos BOUNDS_CHECK for inspiration
        return 0;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION Vector<size_t, LagrangeSpace<T, Dim, Order, ElementType, QuadratureType,
                                                 FieldLHS, FieldRHS>::DeviceStruct::numElementDOFs>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::DeviceStruct::getGlobalDOFIndices(const indices_t& elementNDIndex)
        const {
        Vector<size_t, numElementDOFs> globalDOFs(0);

        // Compute the vector to multiply the ndindex with
        ippl::Vector<size_t, Dim> vec(1);
        for (size_t d = 1; d < dim; ++d) {
            for (size_t d2 = d; d2 < Dim; ++d2) {
                vec[d2] *= this->nr_m[d - 1];
            }
        }
        vec *= Order;  // Multiply each dimension by the order
        size_t smallestGlobalDOF = elementNDIndex.dot(vec);

        // Add the vertex DOFs
        globalDOFs[0] = smallestGlobalDOF;
        globalDOFs[1] = smallestGlobalDOF + Order;

        if constexpr (Dim >= 2) {
            globalDOFs[2] = globalDOFs[1] + this->nr_m[0] * Order;
            globalDOFs[3] = globalDOFs[0] + this->nr_m[0] * Order;
        }
        if constexpr (Dim >= 3) {
            globalDOFs[4] = globalDOFs[0] + this->nr_m[1] * this->nr_m[0] * Order;
            globalDOFs[5] = globalDOFs[1] + this->nr_m[1] * this->nr_m[0] * Order;
            globalDOFs[6] = globalDOFs[2] + this->nr_m[1] * this->nr_m[0] * Order;
            globalDOFs[7] = globalDOFs[3] + this->nr_m[1] * this->nr_m[0] * Order;
        }

        if constexpr (Order > 1) {
            // If the order is greater than 1, there are edge and face DOFs, otherwise the work is
            // done

            // Add the edge DOFs
            if constexpr (Dim >= 2) {
                for (size_t i = 0; i < Order - 1; ++i) {
                    globalDOFs[8 + i]                   = globalDOFs[0] + i + 1;
                    globalDOFs[8 + Order - 1 + i]       = globalDOFs[1] + (i + 1) * this->nr_m[1];
                    globalDOFs[8 + 2 * (Order - 1) + i] = globalDOFs[2] - (i + 1);
                    globalDOFs[8 + 3 * (Order - 1) + i] = globalDOFs[3] - (i + 1) * this->nr_m[1];
                }
            }
            if constexpr (Dim >= 3) {
                // TODO
            }

            // Add the face DOFs
            if constexpr (Dim >= 2) {
                for (size_t i = 0; i < Order - 1; ++i) {
                    for (size_t j = 0; j < Order - 1; ++j) {
                        // TODO CHECK
                        globalDOFs[8 + 4 * (Order - 1) + i * (Order - 1) + j] =
                            globalDOFs[0] + (i + 1) + (j + 1) * this->nr_m[1];
                        globalDOFs[8 + 4 * (Order - 1) + (Order - 1) * (Order - 1) + i * (Order - 1)
                                   + j] = globalDOFs[1] + (i + 1) + (j + 1) * this->nr_m[1];
                        globalDOFs[8 + 4 * (Order - 1) + 2 * (Order - 1) * (Order - 1)
                                   + i * (Order - 1) + j] =
                            globalDOFs[2] - (i + 1) + (j + 1) * this->nr_m[1];
                        globalDOFs[8 + 4 * (Order - 1) + 3 * (Order - 1) * (Order - 1)
                                   + i * (Order - 1) + j] =
                            globalDOFs[3] - (i + 1) + (j + 1) * this->nr_m[1];
                    }
                }
            }
        }

        return globalDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                           FieldRHS>::point_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::DeviceStruct::getRefElementDOFLocation(const size_t& localDOF) const {
        // Assert that the local DOF index is valid
        assert(localDOF < DeviceStruct::numElementDOFs && "The local DOF index is invalid");

        // Use precomputed DOF locations
        return dofLocations_m[localDOF];
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION T
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::DeviceStruct::
        evaluateRefElementShapeFunction(
            const size_t& localDOF,
            const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                FieldRHS>::point_t& localPoint) const {
        // Assert that the local DOF index is valid.
        assert(localDOF < DeviceStruct::numElementDOFs && "The local DOF index is invalid");

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the DOF location on the reference element
        const point_t ref_element_point = getRefElementDOFLocation(localDOF);

        // For Lagrange elements, the shape function is a tensor product of 1D Lagrange polynomials
        T product = 1;

        for (size_t d = 0; d < Dim; d++) {
            // Compute 1D Lagrange basis polynomial in dimension d
            T basis_1d = 1.0;

            // Loop over all nodes in this dimension to construct Lagrange polynomial
            for (unsigned k = 0; k <= Order; ++k) {
                T node_k = static_cast<T>(k) / static_cast<T>(Order);

                // Skip if this is the node corresponding to ref_element_point[d]
                if (Kokkos::abs(ref_element_point[d] - node_k) < 1e-10) {
                    continue;
                }

                // Lagrange basis: product of (x - x_k) / (x_i - x_k)
                basis_1d *= (localPoint[d] - node_k) / (ref_element_point[d] - node_k);
            }

            product *= basis_1d;
        }

        return product;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                           FieldRHS>::indices_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::DeviceStruct::getMeshVertexNDIndex(const size_t& vertex_index) const {
        // Copy the vertex index to the index variable we can alter during the computation.
        size_t index = vertex_index;

        // Create a vector to store the vertex indices in each dimension for the corresponding
        // vertex.
        indices_t vertex_indices;

        // This is the number of vertices in each dimension.
        Vector<size_t, Dim> vertices_per_dim = nr_m;

        // The number_of_lower_dim_vertices is the product of the number of vertices per
        // dimension, it will get divided by the current dimensions number to get the index in
        // that dimension
        size_t remaining_number_of_vertices = 1;
        for (const size_t num_vertices : vertices_per_dim) {
            remaining_number_of_vertices *= num_vertices;
        }

        for (int d = Dim - 1; d >= 0; --d) {
            remaining_number_of_vertices /= vertices_per_dim[d];
            vertex_indices[d] = index / remaining_number_of_vertices;
            index -= vertex_indices[d] * remaining_number_of_vertices;
        }

        return vertex_indices;
    };

}  // namespace ippl
