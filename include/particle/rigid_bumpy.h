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

    Data1D<double> last_angles;  // for tracking particle rotation for vertex neighbor list updates
    Data1D<double> angle_displacements_sq;

    // particle rotational variables
    SwapData1D<double> angles;
    SwapData1D<double> angular_velocities;
    SwapData1D<double> torques;

    SwapData1D<double> area;

    // vertex-based particle variables
    Data1D<long> vertex_particle_index;  // index of the particle that each vertex belongs to
    Data1D<long> particle_start_index;  // index of the first vertex in each particle
    Data1D<long> num_vertices_in_particle;  // number of vertices in each particle

    Data1D<long> vertex_neighbor_list;
    Data1D<long> num_vertex_neighbors;

    double vertex_neighbor_cutoff;  // vertices within this distance of each other are neighbors
    double vertex_particle_neighbor_cutoff;  // particles within this distance of a vertex will be checked for vertex neighbors

    long max_vertex_neighbors_allocated;

    double segment_length_per_vertex_diameter;

    void calculateParticleArea();

    double getParticleArea() const override;

    // sync the vertex radius to the device
    void syncVertexRadius(double vertex_radius);

    double getVertexRadius();

    // vertices should be all the same size and mass?

    void setKernelDimensions(long particle_dim_block = 256, long vertex_dim_block = 256);

    void initVertexVariables();

    void scalePositions(double scale_factor) override;

    void initGeometricVariables();

    void syncVertexIndices();

    void syncVertexNeighborList();

    void initDynamicVariables();
    void clearDynamicVariables();

    void setParticleStartIndex();

    void initializeVerticesFromDiskPacking(SwapData2D<double>& disk_positions, SwapData1D<double>& disk_radii, long num_vertices_in_small_particle);

    double getOverlapFraction() const override;

    void setMass(double mass) override;
    
    void calculateForces() override;

    void updatePositions();

    void updateVelocities();

    void calculateKineticEnergy();

    void calculateParticlePositions();

    void updateVertexVerletList();

    void initVerletListVariables();

    void initVerletList();
};