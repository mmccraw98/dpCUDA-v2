#pragma once

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/data/data_2d.h"
#include "../../include/data/data_1d.h"
#include "particle.h"
#include "config.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <nlohmann/json.hpp>

// vertex sizes are uniform - bidispersity arises from different numbers of vertices per particle
class RigidBumpy : public Particle {
public:
    RigidBumpy();
    virtual ~RigidBumpy();

    SwapData2D<double> vertex_positions;
    SwapData2D<double> vertex_velocities;
    SwapData2D<double> vertex_forces;
    SwapData1D<double> vertex_masses;

    // particle rotational variables
    SwapData1D<double> angles;
    SwapData1D<double> angular_velocities;
    SwapData1D<double> torques;

    SwapData1D<double> area;

    // vertex-based particle variables
    Data1D<long> vertex_particle_index;
    Data1D<long> particle_start_index;
    Data1D<long> num_vertices_in_particle;

    Data1D<long> vertex_neighbor_list;
    Data1D<long> num_vertex_neighbors;

    // Data1D<double> 

    double segment_length_per_vertex_diameter;

    void calculateParticleArea();

    double getParticleArea() const override;

    // sync the vertex radius to the device
    void syncVertexRadius(double vertex_radius);

    // vertices should be all the same size and mass?

    void setKernelDimensions(long particle_dim_block = 256, long vertex_dim_block = 256);

    void initVertexVariables();

    void initGeometricVariables();

    void syncVertexIndices();

    void syncNeighborList() override;

    void initDynamicVariables();
    void clearDynamicVariables();

    void setParticleStartIndex();

    void initializeVerticesFromDiskPacking(SwapData2D<double>& disk_positions, SwapData1D<double>& disk_radii, long num_vertices_in_small_particle);

    double getOverlapFraction() const override;

    void setMass(double mass) override;
    
    void calculateForces() override;

    void calculateKineticEnergy();
};