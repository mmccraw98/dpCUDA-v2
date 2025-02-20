#pragma once

#include <stdio.h>
#include <cmath>
#include "../../../include/constants.h"
#include "../base/kernels.cuh"

// ----------------------------------------------------------------------
// ----------------------- Device Constants -----------------------------
// ----------------------------------------------------------------------

extern __constant__ double d_vertex_radius;

extern __constant__ long* d_num_vertex_neighbors_ptr;
extern __constant__ long* d_vertex_neighbor_list_ptr;
extern __constant__ long d_max_vertex_neighbors_allocated;

extern __constant__ long* d_particle_start_index_ptr;
extern __constant__ long* d_num_vertices_in_particle_ptr;
extern __constant__ long* d_vertex_particle_index_ptr;


// ----------------------------------------------------------------------
// ----------------------- Dynamics and Updates -------------------------
// ----------------------------------------------------------------------


__global__ void kernelCalculateRigidDampedForces(double* forces_x, double* forces_y, double* torques, const double* velocities_x, const double* velocities_y, const double* angular_velocities, const double damping_coefficient);


__global__ void kernelUpdateRigidPositions(
    double* last_positions_x, double* last_positions_y,
    double* positions_x, double* positions_y, double* angles, double* delta_x, double* delta_y, double* angle_delta, const double* last_neigh_positions_x, const double* last_neigh_positions_y, const double* last_cell_positions_x, const double* last_cell_positions_y, double* neigh_displacements_sq, double* cell_displacements_sq, const double* velocities_x, const double* velocities_y, const double* angular_velocities, const double dt);

__global__ void kernelUpdateRigidVelocities(double* velocities_x, double* velocities_y, double* angular_velocities, const double* forces_x, const double* forces_y, const double* torques, const double* masses, const double* moments_of_inertia, const double dt, bool rotation);

// vertex level
__global__ void kernelTranslateAndRotateVertices1(
    const double* last_positions_x, const double* last_positions_y,
    const double* positions_x, const double* positions_y, double* vertex_positions_x, double* vertex_positions_y, const double* delta_x, const double* delta_y, const double* angle_delta);

// particle level
__global__ void kernelTranslateAndRotateVertices2(const double* positions_x, const double* positions_y, double* vertex_positions_x, double* vertex_positions_y, const double* delta_x, const double* delta_y, const double* angle_delta);

__global__ void kernelZeroRigidBumpyParticleForceAndPotentialEnergy(double* forces_x, double* forces_y, double* torques, double* potential_energy);

__global__ void kernelZeroRigidBumpyVertexForceAndPotentialEnergy(double* vertex_forces_x, double* vertex_forces_y, double* vertex_torques, double* vertex_potential_energy);

__global__ void kernelCalculateTranslationalAndRotationalKineticEnergy(
    const double* __restrict__ velocities_x, const double* __restrict__ velocities_y,
    const double* __restrict__ masses, const double* __restrict__ angular_velocities,
    const double* __restrict__ moments_of_inertia, double* __restrict__ kinetic_energy);


// ----------------------------------------------------------------------
// ------------------------- Force Routines -----------------------------
// ----------------------------------------------------------------------

__global__ void kernelCalcRigidBumpyWallForces(const double* positions_x, const double* positions_y, const double* vertex_positions_x, const double* vertex_positions_y, double* vertex_forces_x, double* vertex_forces_y, double* vertex_torques, double* vertex_potential_energy);

// vertex level
__global__ void kernelCalcRigidBumpyForces1(const double* positions_x, const double* positions_y, const double* vertex_positions_x, const double* vertex_positions_y, double* vertex_forces_x, double* vertex_forces_y, double* vertex_torques, double* vertex_potential_energy);

__global__ void kernelCalcRigidBumpyParticleForces1(const double* vertex_forces_x, const double* vertex_forces_y, const double* vertex_torques, const double* vertex_potential_energy, double* particle_forces_x, double* particle_forces_y, double* particle_torques, double* particle_potential_energy);

// particle level
__global__ void kernelCalcRigidBumpyForces2(const double* positions_x, const double* positions_y, const double* vertex_positions_x, const double* vertex_positions_y, double* particle_forces_x, double* particle_forces_y, double* particle_torques, double* particle_potential_energy);

__global__ void kernelCalcRigidBumpyForceDistancePairs(
    const double* positions_x,
    const double* positions_y,
    const double* vertex_positions_x,
    const double* vertex_positions_y,
    double* potential_pairs,
    double* force_pairs_x,
    double* force_pairs_y,
    double* distance_pairs_x,
    double* distance_pairs_y,
    long* this_pair_id,
    long* other_pair_id,
    double* overlap_pairs,
    double* radsum_pairs,
    const double* radii,
    const long* static_particle_index,
    double* pair_separation_angle, 
    double* angle_pairs_i,
    double* angle_pairs_j,
    long* this_vertex_contact_count,
    const double* angles,
    double* pair_friction_coefficient
);

// ----------------------------------------------------------------------
// --------------------- Contacts and Neighbors -------------------------
// ----------------------------------------------------------------------

__global__ void kernelReorderRigidBumpyParticleData(
    const long* __restrict__ particle_index,
    long* __restrict__ old_to_new_particle_index,
    const long* __restrict__ num_vertices_in_particle,
    long* __restrict__ num_vertices_in_particle_new,
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    double* __restrict__ positions_x_new, double* __restrict__ positions_y_new,
    const double* __restrict__ forces_x, const double* __restrict__ forces_y,
    double* __restrict__ forces_x_new, double* __restrict__ forces_y_new,
    const double* __restrict__ velocities_x, const double* __restrict__ velocities_y,
    double* __restrict__ velocities_x_new, double* __restrict__ velocities_y_new,
    const double* __restrict__ angular_velocities, const double* __restrict__ torques,
    double* __restrict__ angular_velocities_new, double* __restrict__ torques_new,
    const double* __restrict__ masses, const double* __restrict__ radii,
    double* __restrict__ masses_new, double* __restrict__ radii_new,
    const double* __restrict__ moments_of_inertia,
    double* __restrict__ moments_of_inertia_new,
    double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y,
    double* __restrict__ cell_displacements_sq);

__global__ void kernelReorderRigidBumpyVertexData(
    const long* __restrict__ vertex_particle_index,
    long* __restrict__ vertex_particle_index_new,
    const long* __restrict__ old_to_new_particle_index,
    const long* __restrict__ particle_start_index,
    const long* __restrict__ particle_start_index_new,
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    double* __restrict__ vertex_positions_x_new, double* __restrict__ vertex_positions_y_new,
    const long* __restrict__ static_vertex_index,
    long* __restrict__ static_vertex_index_new);

__global__ void kernelUpdateVertexNeighborList(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double cutoff,
    const double particle_cutoff
);

// ----------------------------------------------------------------------
// --------------------------- Minimizers -------------------------------
// ----------------------------------------------------------------------

__global__ void kernelRigidBumpyAdamStep(
    double* __restrict__ last_positions_x, double* __restrict__ last_positions_y,
    double* __restrict__ first_moment_x, double* __restrict__ first_moment_y,
    double* __restrict__ first_moment_angle,
    double* __restrict__ second_moment_x, double* __restrict__ second_moment_y,
    double* __restrict__ second_moment_angle,
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    double* __restrict__ angles,
    double* __restrict__ delta_x, double* __restrict__ delta_y,
    double* __restrict__ angle_delta,
    const double* __restrict__ forces_x, const double* __restrict__ forces_y,
    const double* __restrict__ torques,
    double alpha, double beta1, double beta2, double one_minus_beta1_pow_t, double one_minus_beta2_pow_t, double epsilon,
    double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y, double* __restrict__ neigh_displacements_sq,
    double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y, double* __restrict__ cell_displacements_sq, bool rotation);

__global__ void kernelRigidBumpyGradDescStep(
    double* __restrict__ last_positions_x, double* __restrict__ last_positions_y,
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    double* __restrict__ angles,
    double* __restrict__ delta_x, double* __restrict__ delta_y,
    double* __restrict__ angle_delta,
    const double* __restrict__ torques,
    double* __restrict__ forces_x, double* __restrict__ forces_y,
    double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y, double* __restrict__ neigh_displacements_sq,
    double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y, double* __restrict__ cell_displacements_sq,
    double alpha, bool rotation);

// ----------------------------------------------------------------------
// ------------------------ Vertex Utilities ----------------------------
// ----------------------------------------------------------------------

__global__ void kernelSetVertexParticleIndex(
    const long* __restrict__ num_vertices_in_particle,
    const long* __restrict__ particle_start_index,
    long* __restrict__ vertex_particle_index
);

__global__ void kernelGetNumVerticesInParticles(
    const double* __restrict__ radii,
    const double min_particle_diam,
    const long num_vertices_in_small_particle,
    const double max_particle_diam,
    const long num_vertices_in_large_particle,
    long* __restrict__ num_vertices_in_particle);

__global__ void kernelInitializeVerticesOnParticles(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double* __restrict__ radii, const double* __restrict__ angles,
    const long* __restrict__ particle_start_index,
    const long* __restrict__ num_vertices_in_particle,
    double* __restrict__ vertex_positions_x, double* __restrict__ vertex_positions_y);

__global__ void kernelGetVertexMasses(const double* __restrict__ radii, double* __restrict__ vertex_masses, const double* __restrict__ particle_masses);

__global__ void kernelGetMomentsOfInertia(const double* __restrict__ positions_x, const double* __restrict__ positions_y, const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y, const double* __restrict__ vertex_masses, double* __restrict__ moments_of_inertia);

inline __device__ long getNextVertexId(const long current_id, const long num_vertices_in_particle) {
    return mod(current_id + 1, num_vertices_in_particle);
}

inline __device__ long getPreviousVertexId(const long current_id, const long num_vertices_in_particle) {
    return mod(current_id - 1, num_vertices_in_particle);
}

// ----------------------------------------------------------------------
// ---------------------------- Geometry --------------------------------
// ----------------------------------------------------------------------

__global__ void kernelCalculateParticlePolygonArea(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    double* __restrict__ particle_area
);

__global__ void kernelCalculateBumpyParticleAreaFull(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    const double* __restrict__ vertex_radii,
    double* __restrict__ particle_area
);

__global__ void kernelCalculateParticlePositions(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    double* __restrict__ particle_positions_x, double* __restrict__ particle_positions_y
);

__global__ void kernelScalePositions(
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    double* __restrict__ vertex_positions_x, double* __restrict__ vertex_positions_y,
    const double scale_factor
);