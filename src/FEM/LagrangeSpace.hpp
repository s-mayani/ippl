
namespace ippl {
    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::LagrangeSpace(
        const Mesh<T, Dim>& mesh,
        const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::ElementType&
            ref_element,
        const QuadratureType& quadrature)
        : FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), QuadratureType,
                             FieldLHS, FieldRHS>(mesh, ref_element, quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");

        // Initialize the elementIndices view
        initializeElementIndices();
    }

    // Initialize element indices Kokkos View
    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    void LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::initializeElementIndices() {
        const std::size_t numElements = this->numElements();
        elementIndices = Kokkos::View<size_t*>("i", numElements);

        using exec_space  = typename Kokkos::View<size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;
        
        Kokkos::parallel_for("Element index view", policy_type(0, numElements),
            KOKKOS_CLASS_LAMBDA(const size_t i) {
                elementIndices(i) = i;
            });
        Kokkos::fence();
    }

    ///////////////////////////////////////////////////////////////////////
    /// Degree of Freedom operations //////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    std::size_t LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::numGlobalDOFs()
        const {
        std::size_t num_global_dofs = 1;
        for (std::size_t d = 0; d < Dim; ++d) {
            num_global_dofs *= (this->mesh_m.getGridsize(d)) * Order;
        }

        return num_global_dofs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    typename LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t
    LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::getLocalDOFIndex(
        const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t&
            elementIndex,
        const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t&
            globalDOFIndex) const {
        static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");
        // TODO fix not order independent, only works for order 1
        static_assert(Order == 1, "Only order 1 is supported at the moment");

        // Get all the global DOFs for the element
        const Vector<index_t, this->numElementDOFs> global_dofs =
            this->getGlobalDOFIndices(elementIndex);

        ippl::Vector<index_t, this->numElementDOFs> dof_mapping;
        if (Dim == 1) {
            dof_mapping = {0, 1};
        } else if (Dim == 2) {
            dof_mapping = {0, 1, 3, 2};
        } else if (Dim == 3) {
            dof_mapping = {0, 1, 3, 2, 4, 5, 7, 6};
        } else {
            // throw exception
            throw std::runtime_error("FEM Lagrange Space: Dimension not supported: "
                                     + std::to_string(Dim));
        }

        // Find the global DOF in the vector and return the local DOF index
        // TODO this can be done faster since the global DOFs are sorted
        for (index_t i = 0; i < dof_mapping.dim; ++i) {
            if (global_dofs[dof_mapping[i]] == globalDOFIndex) {
                return dof_mapping[i];
            }
        }
        throw std::runtime_error("FEM Lagrange Space: Global DOF not found in specified element");
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    typename LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t
    LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::getGlobalDOFIndex(
        const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t&
            elementIndex,
        const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t&
            localDOFIndex) const {
        const auto global_dofs = this->getGlobalDOFIndices(elementIndex);

        return global_dofs[localDOFIndex];
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    Vector<typename LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t,
           LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::getLocalDOFIndices() const {
        Vector<index_t, numElementDOFs> localDOFs;

        for (index_t dof = 0; dof < numElementDOFs; ++dof) {
            localDOFs[dof] = dof;
        }

        return localDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    Vector<typename LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t,
           LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::getGlobalDOFIndices(
        const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t&
            elementIndex) const {
        Vector<index_t, this->numElementDOFs> globalDOFs(0);

        // get element pos
        ndindex_t elementPos = this->getElementNDIndex(elementIndex);

        // Compute the vector to multiply the ndindex with
        ippl::Vector<std::size_t, Dim> vec(1);
        for (std::size_t d = 1; d < dim; ++d) {
            for (std::size_t d2 = d; d2 < Dim; ++d2) {
                vec[d2] *= this->mesh_m.getGridsize(d - 1);
            }
        }
        vec *= Order;  // Multiply each dimension by the order
        index_t smallestGlobalDOF = elementPos.dot(vec);

        // Add the vertex DOFs
        globalDOFs[0] = smallestGlobalDOF;
        globalDOFs[1] = smallestGlobalDOF + Order;

        if (Dim >= 2) {
            globalDOFs[2] = globalDOFs[1] + this->mesh_m.getGridsize(1) * Order;
            globalDOFs[3] = globalDOFs[0] + this->mesh_m.getGridsize(1) * Order;
        }
        if (Dim >= 3) {
            globalDOFs[4] =
                globalDOFs[0] + this->mesh_m.getGridsize(1) * this->mesh_m.getGridsize(2) * Order;
            globalDOFs[5] =
                globalDOFs[1] + this->mesh_m.getGridsize(1) * this->mesh_m.getGridsize(2) * Order;
            globalDOFs[6] =
                globalDOFs[2] + this->mesh_m.getGridsize(1) * this->mesh_m.getGridsize(2) * Order;
            globalDOFs[7] =
                globalDOFs[3] + this->mesh_m.getGridsize(1) * this->mesh_m.getGridsize(2) * Order;
        }

        if (Order > 1) {
            // If the order is greater than 1, there are edge and face DOFs, otherwise the work is
            // done

            // Add the edge DOFs
            if (Dim >= 2) {
                for (index_t i = 0; i < Order - 1; ++i) {
                    globalDOFs[8 + i] = globalDOFs[0] + i + 1;
                    globalDOFs[8 + Order - 1 + i] =
                        globalDOFs[1] + (i + 1) * this->mesh_m.getGridsize(1);
                    globalDOFs[8 + 2 * (Order - 1) + i] = globalDOFs[2] - (i + 1);
                    globalDOFs[8 + 3 * (Order - 1) + i] =
                        globalDOFs[3] - (i + 1) * this->mesh_m.getGridsize(1);
                }
            }
            if (Dim >= 3) {
                // TODO
            }

            // Add the face DOFs
            if (Dim >= 2) {
                for (index_t i = 0; i < Order - 1; ++i) {
                    for (index_t j = 0; j < Order - 1; ++j) {
                        // TODO CHECK
                        globalDOFs[8 + 4 * (Order - 1) + i * (Order - 1) + j] =
                            globalDOFs[0] + (i + 1) + (j + 1) * this->mesh_m.getGridsize(1);
                        globalDOFs[8 + 4 * (Order - 1) + (Order - 1) * (Order - 1) + i * (Order - 1)
                                   + j] =
                            globalDOFs[1] + (i + 1) + (j + 1) * this->mesh_m.getGridsize(1);
                        globalDOFs[8 + 4 * (Order - 1) + 2 * (Order - 1) * (Order - 1)
                                   + i * (Order - 1) + j] =
                            globalDOFs[2] - (i + 1) + (j + 1) * this->mesh_m.getGridsize(1);
                        globalDOFs[8 + 4 * (Order - 1) + 3 * (Order - 1) * (Order - 1)
                                   + i * (Order - 1) + j] =
                            globalDOFs[3] - (i + 1) + (j + 1) * this->mesh_m.getGridsize(1);
                    }
                }
            }
        }

        return globalDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    Vector<typename LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::ndindex_t,
           LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::getGlobalDOFNDIndices(
        const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t&
            elementIndex) const {
        static_assert(Order == 1 && "Only order 1 is supported at the moment");

        Vector<ndindex_t, numElementDOFs> ndindices;

        // 1. get all the global DOFs for the element
        Vector<index_t, numElementDOFs> global_dofs = this->getGlobalDOFIndices(elementIndex);

        // 2. convert the global DOFs to ndindices
        for (index_t i = 0; i < numElementDOFs; ++i) {
            ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);  // TODO fix for higher order
        }

        return ndindices;
    }

    ///////////////////////////////////////////////////////////////////////
    /// Basis functions and gradients /////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    T LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::
        evaluateRefElementShapeFunction(const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS,
                                                            FieldRHS>::index_t& localDOF,
                                        const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS,
                                                            FieldRHS>::point_t& localPoint) const {
        static_assert(Order == 1, "Only order 1 is supported at the moment");
        // Assert that the local vertex index is valid.
        assert(localDOF < this->numElementDOFs
               && "The local vertex index is invalid");  // TODO assumes 1st order Lagrange

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the local vertex indices for the local vertex index.
        // TODO fix not order independent, only works for order 1
        const point_t ref_element_point = this->ref_element_m.getLocalVertices()[localDOF];

        // The variable that accumulates the product of the shape functions.
        T product = 1;

        for (index_t d = 0; d < Dim; d++) {
            if (localPoint[d] < ref_element_point[d]) {
                product *= localPoint[d];
            } else {
                product *= 1.0 - localPoint[d];
            }
        }

        return product;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    typename LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::gradient_vec_t
    LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::
        evaluateRefElementShapeFunctionGradient(
            const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t&
                localDOF,
            const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::point_t&
                localPoint) const {
        // TODO fix not order independent, only works for order 1
        static_assert(Order == 1 && "Only order 1 is supported at the moment");

        // Assert that the local vertex index is valid.
        assert(localDOF < this->numElementDOFs && "The local vertex index is invalid");

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the local dof nd_index
        const mesh_element_vertex_point_vec_t local_vertex_points =
            this->ref_element_m.getLocalVertices();

        const point_t& local_vertex_point = local_vertex_points[localDOF];

        gradient_vec_t gradient(1);

        // To construct the gradient we need to loop over the dimensions and multiply the
        // shape functions in each dimension except the current one. The one of the current
        // dimension is replaced by the derivative of the shape function in that dimension,
        // which is either 1 or -1.
        for (index_t d = 0; d < Dim; d++) {
            // The variable that accumulates the product of the shape functions.
            T product = 1;

            for (index_t d2 = 0; d2 < Dim; d2++) {
                if (d2 == d) {
                    if (localPoint[d] < local_vertex_point[d]) {
                        product *= 1;
                    } else {
                        product *= -1;
                    }
                } else {
                    if (localPoint[d2] < local_vertex_point[d2]) {
                        product *= localPoint[d2];
                    } else {
                        product *= 1.0 - localPoint[d2];
                    }
                }
            }

            gradient[d] = product;
        }

        return gradient;
    }

    ///////////////////////////////////////////////////////////////////////
    /// Assembly operations ///////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    FieldLHS LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::evaluateAx(
        const FieldLHS& field,
        const std::function<T(
            const index_t&, const index_t&,
            const Vector<Vector<T, Dim>, LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS,
                                                       FieldRHS>::numElementDOFs>&)>& evalFunction)
        const {

        const std::size_t numGhosts = field.getNghost();
        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), numGhosts);

        bool checkEssentialBDCs = true;  // TODO get from field in the future
        // T bc_const_value        = 1.0;   // TODO get from field (non-homogeneous BCs)

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<gradient_vec_t, this->numElementDOFs>, QuadratureType::numElementNodes>
            grad_b_q;
        for (index_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (index_t i = 0; i < this->numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        const std::size_t numElements = this->numElements();
        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);


        Kokkos::parallel_for("Outer loop over elements", policy_type(0, numElements),
            KOKKOS_CLASS_LAMBDA(const size_t index) {  //(index_t elementIndex = 0; elementIndex < numElements; ++elementIndex) {
                const index_t elementIndex                                         = elementIndices(index);
                const Vector<index_t, this->numElementDOFs> local_dofs             = this->getLocalDOFIndices();
                const Vector<ndindex_t, this->numElementDOFs> global_dof_ndindices = this->getGlobalDOFNDIndices(elementIndex);

                // local DOF indices
                index_t i, j;

                // Element matrix
                Vector<Vector<T, this->numElementDOFs>, this->numElementDOFs> A_K;

                // 1. Compute the Galerkin element matrix A_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    for (j = 0; j < this->numElementDOFs; ++j) {
                        A_K[i][j] = 0.0;
                        for (index_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                            A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                        }
                    }
                }

                // global DOF n-dimensional indices (Vector of N indices representing indices in each
                // dimension)
                ndindex_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Skip boundary DOFs (Zero Dirichlet BCs)
                    if (checkEssentialBDCs && this->isDOFOnBoundary(I_nd)) {
                        continue;
                    }

                    for (j = 0; j < this->numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        // Skip boundary DOFs (Zero Dirichlet BCs)
                        if (checkEssentialBDCs && this->isDOFOnBoundary(J_nd)) {
                            continue;
                        }

                        getFieldEntry(resultField, I_nd) += A_K[i][j] * getFieldEntry(field, J_nd);
                    }
                }
        });
        Kokkos::fence();

        IpplTimings::stopTimer(outer_loop);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    void LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::evaluateLoadVector(
        FieldRHS& field, const std::function<T(const point_t&)>& f) const {
        
        const std::size_t numElements = this->numElements();
        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        //Vector<T, this->numElementDOFs> b_K;

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        const ndindex_t zeroNdIndex = Vector<index_t, Dim>(0);

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (index_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (index_t i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        // Absolute value of det Phi_K
        const T absDetDPhi = std::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // TODO move eval function outside of evaluateLoadVector
        const auto eval = [this, absDetDPhi, f](const index_t elementIndex, const index_t& i,
                                                const point_t& q_k,
                                                const Vector<T, this->numElementDOFs>& basis_q_k) {
            const T& f_q_k = f(this->ref_element_m.localToGlobal(
                this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)), q_k));

            return f_q_k * basis_q_k[i] * absDetDPhi;
        };

        Kokkos::parallel_for("Outer loop over elements", policy_type(0, numElements),
            KOKKOS_CLASS_LAMBDA(size_t index) {  //(index_t elementIndex = 0; elementIndex < numElements; ++elementIndex) {
                const index_t elementIndex                              = elementIndices(index);
                const Vector<index_t, this->numElementDOFs> local_dofs  = this->getLocalDOFIndices();
                const Vector<index_t, this->numElementDOFs> global_dofs = this->getGlobalDOFIndices(elementIndex);

                index_t i, I;

                // 1. Compute b_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    I = global_dofs[i];
                    // TODO fix for higher order
                    const auto& dof_ndindex_I = this->getMeshVertexNDIndex(I);

                    if (this->isDOFOnBoundary(dof_ndindex_I)) {
                        continue;
                    }

                    T& b_I = getFieldEntry(field, dof_ndindex_I);

                    for (index_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                        b_I += w[k] * eval(elementIndex, i, q[k], basis_q[k]);
                    }
            }
        });
    }

}  // namespace ippl
