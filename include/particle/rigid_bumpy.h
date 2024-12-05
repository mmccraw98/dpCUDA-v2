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


struct BidisperseRigidBumpyConfig : public BidisperseVertexParticleConfig {

    BidisperseRigidBumpyConfig(
        long seed, long n_particles, double mass, double e_c, double n_c, 
        double packing_fraction, double neighbor_cutoff_multiplier, double neighbor_displacement_multiplier, 
        double num_particles_per_cell, double cell_displacement_multiplier, std::string neighbor_list_update_method, 
        long particle_dim_block,
        long n_vertices, long vertex_dim_block, double vertex_neighbor_cutoff_multiplier, 
        double vertex_neighbor_displacement_multiplier, double segment_length_per_vertex_diameter, bool rotation, double vertex_radius,
        double size_ratio, double count_ratio, long n_vertex_per_small_particle, long n_vertex_per_large_particle
    ) : BidisperseVertexParticleConfig(
        seed, n_particles, mass, e_c, n_c, packing_fraction, 
        neighbor_cutoff_multiplier, neighbor_displacement_multiplier, 
        num_particles_per_cell, cell_displacement_multiplier, neighbor_list_update_method, 
        particle_dim_block, n_vertices, vertex_dim_block, 
        vertex_neighbor_cutoff_multiplier, vertex_neighbor_displacement_multiplier, 
        segment_length_per_vertex_diameter, rotation, vertex_radius,
        size_ratio, count_ratio, n_vertex_per_small_particle, n_vertex_per_large_particle
    ) {
        type_name = "RigidBumpy";
    }
};




// vertex sizes are uniform - bidispersity arises from different numbers of vertices per particle
class RigidBumpy : public Particle {
public:
    RigidBumpy();
    virtual ~RigidBumpy();

    SwapData2D<double> vertex_positions;
    SwapData2D<double> vertex_velocities;
    SwapData2D<double> vertex_forces;
    SwapData1D<double> vertex_torques;
    SwapData1D<double> vertex_masses;
    SwapData1D<double> vertex_potential_energy;
    SwapData1D<double> moments_of_inertia;

    Data1D<double> angle_delta;  // for tracking particle rotation for vertex neighbor list updates
    Data2D<double> delta;  // for tracking particle translation for vertex neighbor list updates
    
    // particle rotational variables
    SwapData1D<double> angles;
    SwapData1D<double> angular_velocities;
    SwapData1D<double> torques;

    SwapData1D<double> area;

    // vertex-based particle variables
    Data1D<long> inverse_particle_index;
    Data1D<long> old_to_new_particle_index;
    SwapData1D<long> vertex_particle_index;  // index of the particle that each vertex belongs to (n_vertices, 1)
    SwapData1D<long> particle_start_index;  // index of the first vertex in each particle (n_particles, 1)
    SwapData1D<long> num_vertices_in_particle;  // number of vertices in each particle (n_particles, 1)

    Data1D<long> vertex_neighbor_list;
    Data1D<long> num_vertex_neighbors;

    SwapData1D<long> vertex_index;  // TODO: probably remove this
    SwapData1D<long> static_vertex_index;

    bool rotation = true;

    double vertex_neighbor_cutoff;  // vertices within this distance of each other are neighbors
    double vertex_particle_neighbor_cutoff;  // particles within this distance of a vertex will be checked for vertex neighbors

    long max_vertex_neighbors_allocated;

    double segment_length_per_vertex_diameter;

    ArrayData getArrayData(const std::string& array_name) override;

    void calculateParticleArea();

    double getParticleArea() const override;

    // sync the vertex radius to the device
    void syncVertexRadius(double vertex_radius);

    double getVertexRadius();

    // vertices should be all the same size and mass?

    long setVertexBiDispersity(long num_vertices_in_small_particle);

    void setKernelDimensions(long particle_dim_block = 256, long vertex_dim_block = 256);

    void initVertexVariables();

    void scalePositions(double scale_factor) override;

    void initGeometricVariables();

    void scaleVelocitiesToTemperature(double temperature) override;
    void setRandomVelocities(double temperature) override;

    void syncVertexIndices();

    void setMomentsOfInertia();

    void syncVertexNeighborList();

    void initDynamicVariables();
    void clearDynamicVariables();

    void setParticleStartIndex();

    void initializeVerticesFromDiskPacking(SwapData2D<double>& disk_positions, SwapData1D<double>& disk_radii, long num_vertices_in_small_particle, long particle_dim_block, long vertex_dim_block);

    double getOverlapFraction() const override;

    void setMass(double mass) override;

    void setDegreesOfFreedom() override;
    
    void calculateForces() override;

    void updatePositions(double dt) override;

    void updateVelocities(double dt) override;

    void calculateKineticEnergy();

    void calculateParticlePositions();

    void updateVertexVerletList();

    double getGeometryScale() override;

    void initVerletListVariables() override;

    void initVerletList() override;

    void updateVerletList() override;

    bool setNeighborSize(double neighbor_cutoff_multiplier, double neighbor_displacement_multiplier) override;

    void reorderParticleData() override;

    void initCellListVariables() override;

    void updateCellNeighborList() override;

    void zeroForceAndPotentialEnergy() override;

    void initializeFromConfig(BidisperseRigidBumpyConfig& config);

    void initCellList() override;
};