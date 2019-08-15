// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2019 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Nic Olsen, Dan Negrut
// =============================================================================

#include <string>
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/utils/ChGranularUtilities.h"
#include "chrono_granular/physics/ChGranularTriMesh.h"

void ChGranularChronoTriMeshAPI::load_meshes(std::vector<std::string> objfilenames,
                                             std::vector<chrono::ChMatrix33<float>> rotscale,
                                             std::vector<float3> translations,
                                             std::vector<float> masses,
                                             std::vector<bool> inflated,
                                             std::vector<float> inflation_radii) {
    unsigned int size = objfilenames.size();
    if (size != rotscale.size() || size != translations.size() || size != masses.size() || size != inflated.size() ||
        size != inflation_radii.size()) {
        GRANULAR_ERROR("Mesh loading vectors must all have same size\n");
    }

    if (size == 0) {
        printf("WARNING: No meshes provided!\n");
    }

    unsigned int nTriangles = 0;
    unsigned int numTriangleFamilies = 0;
    std::vector<chrono::geometry::ChTriangleMeshConnected> all_meshes;
    for (unsigned int i = 0; i < objfilenames.size(); i++) {
        MESH_INFO_PRINTF("Importing %s...\n", objfilenames[i].c_str());
        all_meshes.push_back(chrono::geometry::ChTriangleMeshConnected());
        chrono::geometry::ChTriangleMeshConnected& mesh = all_meshes[all_meshes.size() - 1];

        mesh.LoadWavefrontMesh(objfilenames[i], true, false);

        // Apply displacement
        chrono::ChVector<> displ(translations[i].x, translations[i].y, translations[i].z);

        // Apply scaling and then rotation
        mesh.Transform(displ, rotscale[i].cast<double>());

        unsigned int num_triangles_curr = mesh.getNumTriangles();

        if (num_triangles_curr == 0) {
            GRANULAR_ERROR("ERROR! Mesh %s has no triangles in it! Exiting!\n", objfilenames[i].c_str());
        }

        nTriangles += num_triangles_curr;
        numTriangleFamilies++;
    }

    MESH_INFO_PRINTF("nTriangles is %u\n", nTriangles);
    MESH_INFO_PRINTF("nTriangleFamiliesInSoup is %u\n", numTriangleFamilies);

    // Allocate memory to store mesh soup in unified memory
    MESH_INFO_PRINTF("Allocating mesh unified memory\n");
    setupTriMesh(all_meshes, nTriangles, masses, inflated, inflation_radii);
    MESH_INFO_PRINTF("Done allocating mesh unified memory\n");
}

void ChGranularChronoTriMeshAPI::setupTriMesh(const std::vector<chrono::geometry::ChTriangleMeshConnected>& all_meshes,
                                              unsigned int nTriangles,
                                              std::vector<float> masses,
                                              std::vector<bool> inflated,
                                              std::vector<float> inflation_radii) {
    meshSoup->nTrianglesInSoup = nTriangles;

    if (nTriangles != 0) {
        // Allocate all of the requisite pointers
        gpuErrchk(
            cudaMallocManaged(&meshSoup->triangleFamily_ID, nTriangles * sizeof(unsigned int), cudaMemAttachGlobal));

        gpuErrchk(cudaMallocManaged(&meshSoup->node1, nTriangles * sizeof(float3), cudaMemAttachGlobal));
        gpuErrchk(cudaMallocManaged(&meshSoup->node2, nTriangles * sizeof(float3), cudaMemAttachGlobal));
        gpuErrchk(cudaMallocManaged(&meshSoup->node3, nTriangles * sizeof(float3), cudaMemAttachGlobal));
    }

    MESH_INFO_PRINTF("Done allocating nodes for %d triangles\n", nTriangles);

    // Setup the clean copy of the mesh soup from the obj file data
    unsigned int family = 0;
    unsigned int tri_i = 0;
    // for each obj file data set
    for (auto mesh : all_meshes) {
        int n_triangles_mesh = mesh.getNumTriangles();
        for (int i = 0; i < n_triangles_mesh; i++) {
            chrono::geometry::ChTriangle tri = mesh.getTriangle(i);

            meshSoup->node1[tri_i] = make_float3(tri.p1.x(), tri.p1.y(), tri.p1.z());
            meshSoup->node2[tri_i] = make_float3(tri.p2.x(), tri.p2.y(), tri.p2.z());
            meshSoup->node3[tri_i] = make_float3(tri.p3.x(), tri.p3.y(), tri.p3.z());

            meshSoup->triangleFamily_ID[tri_i] = family;

            // Normal of a single vertex... Should still work
            int normal_i = mesh.m_face_n_indices.at(i).x();  // normals at each vertex of this triangle
            chrono::ChVector<double> normal = mesh.m_normals[normal_i];

            // Generate normal using RHR from nodes 1, 2, and 3
            chrono::ChVector<double> AB = tri.p2 - tri.p1;
            chrono::ChVector<double> AC = tri.p3 - tri.p1;
            chrono::ChVector<double> cross;
            cross.Cross(AB, AC);

            // If the normal created by a RHR traversal is not correct, switch two vertices
            if (cross.Dot(normal) < 0) {
                std::swap(meshSoup->node2[tri_i], meshSoup->node3[tri_i]);
            }
            tri_i++;
        }
        family++;
        MESH_INFO_PRINTF("Done writing family %d\n", family);
    }

    meshSoup->numTriangleFamilies = family;

    if (meshSoup->nTrianglesInSoup != 0) {
        gpuErrchk(cudaMallocManaged(&meshSoup->familyMass_SU, family * sizeof(float), cudaMemAttachGlobal));
        gpuErrchk(cudaMallocManaged(&meshSoup->inflated, family * sizeof(float), cudaMemAttachGlobal));
        gpuErrchk(cudaMallocManaged(&meshSoup->inflation_radii, family * sizeof(float), cudaMemAttachGlobal));

        for (unsigned int i = 0; i < family; i++) {
            // NOTE The SU conversion is done in initialize after the scaling is determined
            meshSoup->familyMass_SU[i] = masses[i];
            meshSoup->inflated[i] = inflated[i];
            meshSoup->inflation_radii[i] = inflation_radii[i];
        }

        gpuErrchk(cudaMallocManaged(&meshSoup->generalizedForcesPerFamily,
                                    6 * meshSoup->numTriangleFamilies * sizeof(float), cudaMemAttachGlobal));
        // Allocate memory for the float and double frames
        gpuErrchk(
            cudaMallocManaged(&tri_params->fam_frame_broad,
                              meshSoup->numTriangleFamilies * sizeof(chrono::granular::ChGranMeshFamilyFrame<float>),
                              cudaMemAttachGlobal));
        gpuErrchk(
            cudaMallocManaged(&tri_params->fam_frame_narrow,
                              meshSoup->numTriangleFamilies * sizeof(chrono::granular::ChGranMeshFamilyFrame<double>),
                              cudaMemAttachGlobal));

        // Allocate memory for linear and angular velocity
        gpuErrchk(
            cudaMallocManaged(&meshSoup->vel, meshSoup->numTriangleFamilies * sizeof(float3), cudaMemAttachGlobal));
        gpuErrchk(
            cudaMallocManaged(&meshSoup->omega, meshSoup->numTriangleFamilies * sizeof(float3), cudaMemAttachGlobal));

        for (unsigned int i = 0; i < family; i++) {
            meshSoup->vel[i] = make_float3(0, 0, 0);
            meshSoup->omega[i] = make_float3(0, 0, 0);
        }
    }
}

void ChGranularSMC_API::setElemsPositions(const std::vector<chrono::ChVector<float>>& points) {
    std::vector<float3> pointsFloat3;
    pSMCgranSystem->setParticlePositions(pointsFloat3);
}

