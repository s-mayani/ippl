// DOFHandler implementation file
//   This file contains the implementation of the DOFHandler class methods.

#ifndef IPPL_DOFHANDLER_HPP
#define IPPL_DOFHANDLER_HPP

namespace ippl {

    template <typename T, typename SpaceTraits_>
    DOFHandler<T, SpaceTraits_>::DOFHandler()
        : mesh_m(nullptr) {
        // Default constructor
    }

    template <typename T, typename SpaceTraits_>
    DOFHandler<T, SpaceTraits_>::DOFHandler(Mesh_t& mesh, const Layout_t& layout) : mesh_m(&mesh) {
        initialize(mesh, layout);
    }

    template <typename T, typename SpaceTraits_>
    void DOFHandler<T, SpaceTraits_>::initialize(Mesh_t& mesh, const Layout_t& layout) {
        mesh_m = &mesh;

        // Get number of elements in each direction
        for (unsigned d = 0; d < Dim; ++d) {
            ne_m[d] = mesh.getGridsize()[d] - 1;
        }

        // Create element layout (one less than vertices in each dimension)
        NDIndex<Dim> elementDomain = layout.getDomain();
        for (unsigned d = 0; d < Dim; ++d) {
            elementDomain[d] = elementDomain[d].cut(1);
        }
        auto elementLayout = SubFieldLayout<Dim>(layout.comm, layout.getDomain(),
                                               elementDomain, layout.isParallel(),
                                               layout.isAllPeriodic_m);

        lElemDom_m = elementLayout.getLocalNDIndex();

        // Allocate device view for DOF mapping table
        dofMappingTable_m = Kokkos::View<DOFMapping*>("DOFMappingTable", dofsPerElement);

        // Create host mirror and fill the mapping table on host
        dofMappingTable_h = Kokkos::create_mirror_view(dofMappingTable_m);
        if constexpr (std::is_same_v<typename SpaceTraits::SpaceTag, LagrangeSpaceTag>) {
            fillLagrangeDOFMappingTable(dofMappingTable_h);
        } else if constexpr (std::is_same_v<typename SpaceTraits::SpaceTag, NedelecSpaceTag>) {
            fillNedelecDOFMappingTable(dofMappingTable_h);
        }

        // Deep copy from host to device
        Kokkos::deep_copy(dofMappingTable_m, dofMappingTable_h);
    }
    
    template <typename T, typename SpaceTraits_>
    void DOFHandler<T, SpaceTraits_>::fillLagrangeDOFMappingTable(typename Kokkos::View<DOFMapping*>::HostMirror& hostTable) const {
        size_t dofIndex = 0;

        // For Lagrange elements, DOFs are ordered as:
        // vertices first, then edges (X, Y, Z), then faces (XY, XZ, YZ), then volume
        // All numbering follows counter-clockwise convention

        // Helper arrays for counter-clockwise numbering
        // 2D vertices (counter-clockwise starting bottom-left): 0:[0,0], 1:[1,0], 2:[1,1], 3:[0,1]
        constexpr Kokkos::Array<Kokkos::Array<size_t, 2>, 4> vertex2DOffsets = {{
            {0, 0}, {1, 0}, {1, 1}, {0, 1}
        }};

        // 3D vertices (counter-clockwise in XY at z=0, then z=1)
        constexpr Kokkos::Array<Kokkos::Array<size_t, 3>, 8> vertex3DOffsets = {{
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},  // z=0 plane
            {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}   // z=1 plane
        }};

        // 2D EdgeX (horizontal, counter-clockwise): 0:[0,0], 1:[0,1]
        constexpr Kokkos::Array<Kokkos::Array<size_t, 2>, 2> edge2DXOffsets = {{
            {0, 0}, {0, 1}
        }};

        // 2D EdgeY (vertical, counter-clockwise): 0:[0,0], 1:[1,0]
        constexpr Kokkos::Array<Kokkos::Array<size_t, 2>, 2> edge2DYOffsets = {{
            {0, 0}, {1, 0}
        }};

        // 3D EdgeX (parallel to X, counter-clockwise in YZ): 0:[0,0,0], 1:[0,1,0], 2:[0,1,1], 3:[0,0,1]
        constexpr Kokkos::Array<Kokkos::Array<size_t, 3>, 4> edge3DXOffsets = {{
            {0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}
        }};

        // 3D EdgeY (parallel to Y, counter-clockwise in XZ): 0:[0,0,0], 1:[1,0,0], 2:[1,0,1], 3:[0,0,1]
        constexpr Kokkos::Array<Kokkos::Array<size_t, 3>, 4> edge3DYOffsets = {{
            {0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}
        }};

        // 3D EdgeZ (parallel to Z, counter-clockwise in XY): 0:[0,0,0], 1:[1,0,0], 2:[1,1,0], 3:[0,1,0]
        constexpr Kokkos::Array<Kokkos::Array<size_t, 3>, 4> edge3DZOffsets = {{
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}
        }};

        // Vertex DOFs
        constexpr size_t verticesPerElement = (1 << Dim); // 2^Dim
        if constexpr (TagIndex<EntityTypes>::template contains<Vertex<Dim>>()) {
            constexpr size_t vertexTypeIndex = TagIndex<EntityTypes>::template index<Vertex<Dim>>();
            constexpr size_t vertexDOFCount = SpaceTraits::template entityDOFCount<Vertex<Dim>>();

            for (size_t v = 0; v < verticesPerElement; ++v) {
                indices_t offset;
                if constexpr (Dim == 1) {
                    offset[0] = v; // Simple: 0 or 1
                } else if constexpr (Dim == 2) {
                    offset[0] = vertex2DOffsets[v][0];
                    offset[1] = vertex2DOffsets[v][1];
                } else if constexpr (Dim == 3) {
                    offset[0] = vertex3DOffsets[v][0];
                    offset[1] = vertex3DOffsets[v][1];
                    offset[2] = vertex3DOffsets[v][2];
                }
                for (size_t localDOF = 0; localDOF < vertexDOFCount; ++localDOF) {
                    hostTable(dofIndex) = {vertexTypeIndex, offset, localDOF};
                    ++dofIndex;
                }
            }
        }

        // Edge DOFs

        // X-oriented edges
        if constexpr (TagIndex<EntityTypes>::template contains<EdgeX<Dim>>()) {
            constexpr size_t edgeXTypeIndex = TagIndex<EntityTypes>::template index<EdgeX<Dim>>();
            constexpr size_t edgeXDOFCount = SpaceTraits::template entityDOFCount<EdgeX<Dim>>();
            constexpr size_t numEdgesX = (Dim == 1) ? 1 : ((Dim == 2) ? 2 : 4);

            for (size_t e = 0; e < numEdgesX; ++e) {
                indices_t offset;
                if constexpr (Dim == 1) {
                    offset[0] = 0;  // In 1D, edge is the element itself at position 0
                } else if constexpr (Dim == 1) {
                    offset[0] = 0;  // In 1D, edge is the element itself at position 0
                } else if constexpr (Dim == 2) {
                    offset[0] = edge2DXOffsets[e][0];
                    offset[1] = edge2DXOffsets[e][1];
                } else if constexpr (Dim == 3) {
                    offset[0] = edge3DXOffsets[e][0];
                    offset[1] = edge3DXOffsets[e][1];
                    offset[2] = edge3DXOffsets[e][2];
                }
                for (size_t localDOF = 0; localDOF < edgeXDOFCount; ++localDOF) {
                    hostTable(dofIndex) = {edgeXTypeIndex, offset, localDOF};
                    ++dofIndex;
                }
            }
        }

        // Y-oriented edges (2D and 3D)
        if constexpr (Dim >= 2) {
            if constexpr (TagIndex<EntityTypes>::template contains<EdgeY<Dim>>()) {
                constexpr size_t edgeYTypeIndex = TagIndex<EntityTypes>::template index<EdgeY<Dim>>();
                constexpr size_t edgeYDOFCount = SpaceTraits::template entityDOFCount<EdgeY<Dim>>();
                constexpr size_t numEdgesY = (Dim == 2) ? 2 : 4;

                for (size_t e = 0; e < numEdgesY; ++e) {
                    indices_t offset;
                    if constexpr (Dim == 2) {
                        offset[0] = edge2DYOffsets[e][0];
                        offset[1] = edge2DYOffsets[e][1];
                    } else if constexpr (Dim == 3) {
                        offset[0] = edge3DYOffsets[e][0];
                        offset[1] = edge3DYOffsets[e][1];
                        offset[2] = edge3DYOffsets[e][2];
                    }
                    for (size_t localDOF = 0; localDOF < edgeYDOFCount; ++localDOF) {
                        hostTable(dofIndex) = {edgeYTypeIndex, offset, localDOF};
                        ++dofIndex;
                    }
                }
            }
        }

        // Z-oriented edges (3D only)
        if constexpr (Dim == 3) {
            if constexpr (TagIndex<EntityTypes>::template contains<EdgeZ<Dim>>()) {
                constexpr size_t edgeZTypeIndex = TagIndex<EntityTypes>::template index<EdgeZ<Dim>>();
                constexpr size_t edgeZDOFCount = SpaceTraits::template entityDOFCount<EdgeZ<Dim>>();

                for (size_t e = 0; e < 4; ++e) {
                    indices_t offset;
                    offset[0] = edge3DZOffsets[e][0];
                    offset[1] = edge3DZOffsets[e][1];
                    offset[2] = edge3DZOffsets[e][2];
                    for (size_t localDOF = 0; localDOF < edgeZDOFCount; ++localDOF) {
                        hostTable(dofIndex) = {edgeZTypeIndex, offset, localDOF};
                        ++dofIndex;
                    }
                }
            }
        }

        // Face DOFs (2D and 3D)
        if constexpr (Dim >= 2) {
            // FaceXY (in 2D: the element itself, in 3D: XY-plane faces)
            if constexpr (TagIndex<EntityTypes>::template contains<FaceXY<Dim>>()) {
                constexpr size_t faceXYTypeIndex = TagIndex<EntityTypes>::template index<FaceXY<Dim>>();
                constexpr size_t faceXYDOFCount = SpaceTraits::template entityDOFCount<FaceXY<Dim>>();
                constexpr size_t numFacesXY = (Dim == 2) ? 1 : 2;

                for (size_t f = 0; f < numFacesXY; ++f) {
                    indices_t offset;
                    offset[0] = 0;
                    offset[1] = 0;
                    if constexpr (Dim == 3) {
                        offset[2] = f; // FaceXY 0:[0,0,0], FaceXY 1:[0,0,1]
                    }
                    for (size_t localDOF = 0; localDOF < faceXYDOFCount; ++localDOF) {
                        hostTable(dofIndex) = {faceXYTypeIndex, offset, localDOF};
                        ++dofIndex;
                    }
                }
            }

            // FaceXZ (3D only)
            if constexpr (Dim == 3) {
                if constexpr (TagIndex<EntityTypes>::template contains<FaceXZ<Dim>>()) {
                    constexpr size_t faceXZTypeIndex = TagIndex<EntityTypes>::template index<FaceXZ<Dim>>();
                    constexpr size_t faceXZDOFCount = SpaceTraits::template entityDOFCount<FaceXZ<Dim>>();

                    for (size_t f = 0; f < 2; ++f) {
                        indices_t offset;
                        offset[0] = 0;
                        offset[1] = f; // FaceXZ 0:[0,0,0], FaceXZ 1:[0,1,0]
                        offset[2] = 0;
                        for (size_t localDOF = 0; localDOF < faceXZDOFCount; ++localDOF) {
                            hostTable(dofIndex) = {faceXZTypeIndex, offset, localDOF};
                            ++dofIndex;
                        }
                    }
                }
            }

            // FaceYZ (3D only)
            if constexpr (Dim == 3) {
                if constexpr (TagIndex<EntityTypes>::template contains<FaceYZ<Dim>>()) {
                    constexpr size_t faceYZTypeIndex = TagIndex<EntityTypes>::template index<FaceYZ<Dim>>();
                    constexpr size_t faceYZDOFCount = SpaceTraits::template entityDOFCount<FaceYZ<Dim>>();

                    for (size_t f = 0; f < 2; ++f) {
                        indices_t offset;
                        offset[0] = f; // FaceYZ 0:[0,0,0], FaceYZ 1:[1,0,0]
                        offset[1] = 0;
                        offset[2] = 0;
                        for (size_t localDOF = 0; localDOF < faceYZDOFCount; ++localDOF) {
                            hostTable(dofIndex) = {faceYZTypeIndex, offset, localDOF};
                            ++dofIndex;
                        }
                    }
                }
            }
        }

        // Volume DOFs (3D only)
        if constexpr (Dim == 3) {
            if constexpr (TagIndex<EntityTypes>::template contains<Hexahedron<Dim>>()) {
                constexpr size_t hexTypeIndex = TagIndex<EntityTypes>::template index<Hexahedron<Dim>>();
                constexpr size_t hexDOFCount = SpaceTraits::template entityDOFCount<Hexahedron<Dim>>();

                indices_t offset;
                offset[0] = 0;
                offset[1] = 0;
                offset[2] = 0;
                for (size_t localDOF = 0; localDOF < hexDOFCount; ++localDOF) {
                    hostTable(dofIndex) = {hexTypeIndex, offset, localDOF};
                    ++dofIndex;
                }
            }
        }
    }
        
    template <typename T, typename SpaceTraits_>
    void DOFHandler<T, SpaceTraits_>::fillNedelecDOFMappingTable(typename Kokkos::View<DOFMapping*>::HostMirror& hostTable) const {
        // TODO implement Nedelec DOF mapping table filling
    }

    template <typename T, typename SpaceTraits_>
    KOKKOS_INLINE_FUNCTION typename DOFHandler<T, SpaceTraits_>::DOFMapping
    DOFHandler<T, SpaceTraits_>::getElementDOFMapping(const size_t& localElementDOF) const {
        // Use Kokkos macros to select host mirror on host, device view on device
        DOFMapping result;
        KOKKOS_IF_ON_DEVICE((result = dofMappingTable_m(localElementDOF);))
        KOKKOS_IF_ON_HOST((result = dofMappingTable_h(localElementDOF);))
        return result;
    }

    template <typename T, typename SpaceTraits_>
    KOKKOS_FUNCTION typename DOFHandler<T, SpaceTraits_>::indices_t
    DOFHandler<T, SpaceTraits_>::getElementNDIndex(const size_t& elementIndex) const {
        // Convert linear element index to NDIndex using row-major ordering
        size_t index = elementIndex;
        indices_t ndIndex;

        size_t remaining_number_of_cells = 1;
        for (unsigned d = 0; d < Dim; ++d) {
            remaining_number_of_cells *= ne_m[d];
        }

        for (int d = Dim - 1; d >= 0; --d) {
            remaining_number_of_cells /= ne_m[d];
            ndIndex[d] = (index / remaining_number_of_cells);
            index -= ndIndex[d] * remaining_number_of_cells;
        }

        return ndIndex;
    }

    template <typename T, typename SpaceTraits_>
    KOKKOS_FUNCTION typename DOFHandler<T, SpaceTraits_>::indices_t
    DOFHandler<T, SpaceTraits_>::getLocalElementNDIndex(const size_t& elementIndex, int nghost) const {
        // Convert linear element index to global NDIndex using row-major ordering
        indices_t ndIndex = getElementNDIndex(elementIndex);

        // Convert to local coordinates by subtracting the local domain offset
        for (unsigned d = 0; d < Dim; ++d) {
            ndIndex[d] = ndIndex[d] - lElemDom_m[d].first() + nghost;
        }

        return ndIndex;
    }

    template <typename T, typename SpaceTraits_>
    KOKKOS_FUNCTION bool
    DOFHandler<T, SpaceTraits_>::isDOFOnBoundary(const size_t& elementIndex, const size_t& localDOF,
                                                   const unsigned& dim) const {
        // Get the DOF mapping to find which entity this DOF belongs to
        DOFMapping dofMap = getElementDOFMapping(localDOF);

        // Check if this entity type can touch boundaries perpendicular to dimension 'dim'
        // An entity can only touch such boundaries if it does NOT extend in dimension 'dim'
        // We need to get entityDir[dim] for the entity type at entityTypeIndex
        bool entityExtendsInDim = false;

        // Unroll at compile time over all entity types
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((dofMap.entityTypeIndex == Is ?
                (entityExtendsInDim = std::tuple_element_t<Is, EntityTypes>{}.getDir()[dim], true) : false), ...);
        }(std::make_index_sequence<numEntityTypes>{});

        // If entity extends in this dimension, it cannot be on a boundary perpendicular to it
        if (entityExtendsInDim) {
            return false;
        }

        // Get the element's global NDIndex
        indices_t elementNDIndex = getElementNDIndex(elementIndex);

        // Calculate the entity's position in the specified dimension
        size_t entityCoord = elementNDIndex[dim] + dofMap.entityLocalIndex[dim];

        // Check if the entity is on the boundary in this dimension
        // An entity is on the boundary if its coordinate equals 0 (lower boundary)
        // or ne_m[dim] (upper boundary)
        return (entityCoord == 0 || entityCoord == ne_m[dim]);
    }

    template <typename T, typename SpaceTraits_>
    KOKKOS_FUNCTION bool
    DOFHandler<T, SpaceTraits_>::isDOFOnBoundary(const size_t& elementIndex, const size_t& localDOF) const {
        bool isOnBoundary = false;

        // Check all dimensions using bitwise OR to avoid branching
        for (unsigned d = 0; d < Dim; ++d) {
            isOnBoundary |= isDOFOnBoundary(elementIndex, localDOF, d);
        }

        return isOnBoundary;
    }

    template <typename T, typename SpaceTraits_>
    Kokkos::View<size_t*> DOFHandler<T, SpaceTraits_>::getElementIndices() const {
        if (mesh_m == nullptr) {
            throw IpplException("DOFHandler::getElementIndices",
                              "DOFHandler not initialized. Call initialize() first.");
        }

        unsigned localElementCount = lElemDom_m.size();

        Kokkos::View<size_t*> localElementIndices("localElementIndices", localElementCount);

        Kokkos::parallel_for(
            "ComputeLocalElementsCount", localElementCount,
            KOKKOS_CLASS_LAMBDA(const int i) {
                int idx = i;
                indices_t ndIndex;

                // Convert linear index to NDIndex within local subdomain
                for (unsigned int d = 0; d < Dim; ++d) {
                    const int range = lElemDom_m[d].last() - lElemDom_m[d].first() + 1;
                    ndIndex[d] = lElemDom_m[d].first() + (idx % range);
                    idx /= range;
                }

                // Convert NDIndex to global element index
                size_t elementIndex = 0;
                size_t multiplier = 1;
                for (unsigned int d = 0; d < Dim; ++d) {
                    elementIndex += ndIndex[d] * multiplier;
                    multiplier *= ne_m[d];
                }

                localElementIndices(i) = elementIndex;
            });

        // Fence to ensure the view is populated before returning
        Kokkos::fence();

        return localElementIndices;
    }

}  // namespace ippl

#endif  // IPPL_DOFHANDLER_HPP
