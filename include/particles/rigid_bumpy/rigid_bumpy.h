#pragma once

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/data/data_2d.h"
#include "../../include/data/data_1d.h"
#include "../../include/particles/base/particle.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <nlohmann/json.hpp>

#include "../../include/routines/minimization.h"

// vertex sizes are uniform - bidispersity arises from different numbers of vertices per particle
class RigidBumpy : public Particle {
public:
    RigidBumpy();
    virtual ~RigidBumpy();

    void initializeFromConfig(ConfigDict& config, bool minimize = false) override;

    long load(std::filesystem::path root_path, std::string source, long frame = -2) override;

    void loadDataFromPath(std::filesystem::path root_path, std::string data_file_extension) override;

    std::vector<std::string> get_reorder_arrays() override { return {"static_particle_index", "static_vertex_index"}; }  // possibly need to replicate for each derived class - tracks the arrays used to index particle level data

    std::vector<std::string> getFundamentalValues() override { return {"radii", "positions", "angles", "velocities", "angular_velocities", "num_vertices_in_particle", "vertex_positions", "masses", "moments_of_inertia", "box_size", "vertex_masses"}; }

    void calculateDiskArea();

    bool tryLoadArrayData(std::filesystem::path path) override;

    SwapData2D<double> vertex_positions;
    SwapData2D<double> vertex_velocities;
    SwapData2D<double> vertex_forces;
    SwapData1D<double> vertex_torques;
    SwapData1D<double> vertex_masses;
    SwapData1D<double> vertex_potential_energy;
    SwapData1D<double> moments_of_inertia;

    Data2D<double> last_positions;

    Data1D<double> first_moment_angle;
    Data1D<double> second_moment_angle;

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

    // "angle_pairs_i", "angle_pairs_j", "pair_separation_angle", "pair_contact_vertex_count"
    Data1D<double> angle_pairs_i;
    Data1D<double> angle_pairs_j;
    Data1D<long> this_vertex_contact_counts;
    Data1D<double> pair_friction_coefficient;
    Data1D<double> pair_vertex_overlaps;

    bool rotation = true;

    void setRotation(bool rotation);

    std::unordered_map<std::string, std::vector<std::string>> calculation_dependencies = {  // replicate this for each derived class
        {"TE", {"calculate_kinetic_energy"}},
        {"T", {"calculate_kinetic_energy"}},
        {"KE", {"calculate_kinetic_energy"}},  // total kinetic energy scalar
        {"Zp", {"count_contacts"}},
        {"P", {"calculate_stress_tensor"}},
        {"stress_tensor_x", {"calculate_stress_tensor"}},
        {"stress_tensor_y", {"calculate_stress_tensor"}},
        {"stress_tensor", {"calculate_stress_tensor"}},
        {"kinetic_energy", {"calculate_kinetic_energy"}},  // kinetic energy array
        {"potential_pairs", {"calculate_force_distance_pairs"}},
        {"force_pairs", {"calculate_force_distance_pairs"}},
        {"distance_pairs", {"calculate_force_distance_pairs"}},
        {"pair_ids", {"calculate_force_distance_pairs"}},
        {"pair_separation_angle", {"calculate_force_distance_pairs"}},
        {"angle_pairs_i", {"calculate_force_distance_pairs"}},
        {"angle_pairs_j", {"calculate_force_distance_pairs"}},
        {"this_vertex_contact_counts", {"calculate_force_distance_pairs"}},
        {"vertex_contact_counts_j", {"calculate_force_distance_pairs"}},
        {"pair_friction_coefficient", {"calculate_force_distance_pairs"}},
        {"pair_vertex_overlaps", {"calculate_force_distance_pairs"}},
        // can have nested dependencies i.e. {"particle_KE", {"calculate_particle_kinetic_energy"}}, {"calculate_particle_kinetic_energy", {"calculate_particle_velocities"}}
    };

    double vertex_neighbor_cutoff;  // vertices within this distance of each other are neighbors
    double vertex_particle_neighbor_cutoff;  // particles within this distance of a vertex will be checked for vertex neighbors

    long max_vertex_neighbors_allocated;

    double segment_length_per_vertex_diameter;

    ArrayData getArrayData(const std::string& array_name) override;

    void calculateParticleArea() override;

    void calculateNumVerticesInParticles(long num_vertices_in_small_particle, long num_vertices_in_large_particle);

    void setRandomAngles();

    void setVertexParticleIndex();

    void setVertexPositions();

    double calculateVertexRadius(long num_vertices_in_small_particle);

    long calculateNumVertices(long num_vertices_in_small_particle, long num_vertices_in_large_particle);

    void setSegmentLengthPerVertexDiameter(double segment_length_per_vertex_diameter);

    double getSegmentLengthPerVertexDiameter();

    void finalizeLoading() override;

    double getParticleArea() const override;

    // sync the vertex radius to the device
    void syncVertexRadius(double vertex_radius);

    double getVertexRadius();

    // vertices should be all the same size and mass?

    long setVertexBiDispersity(long num_vertices_in_small_particle);

    void setKernelDimensions(long particle_dim_block = 256, long vertex_dim_block = 256);

    void initVertexVariables();

    void scalePositions(double scale_factor) override;

    void scalePositionsFull(double scale_factor) override;

    void initGeometricVariables();

    void scaleVelocitiesToTemperature(double temperature) override;
    void setRandomVelocities(double temperature) override;

    void stopRattlerVelocities() override;

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

    void countContacts() override;

    void reorderParticleData() override;

    void initCellListVariables() override;

    void updateCellNeighborList() override;

    void zeroForceAndPotentialEnergy() override;

    void initCellList() override;

    void calculateForceDistancePairs();

    void initAdamVariables() override;
    void clearAdamVariables() override;
    void updatePositionsAdam(long step, double alpha, double beta1, double beta2, double epsilon) override;

    void calculateWallForces();

    void calculateDampedForces(double damping_coefficient);

    void calculateStressTensor() override;

    void loadData(const std::string& root) override;
};