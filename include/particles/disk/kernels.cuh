#pragma once

#include <stdio.h>
#include <cmath>
#include "../../../include/constants.h"
#include "../base/kernels.cuh"

// ----------------------------------------------------------------------
// ------------------------- Force Routines -----------------------------
// ----------------------------------------------------------------------

/**
 * @brief Calculate the interaction forces and energy between a particle and its neighbors.
 * Requires that the potential energy and force arrays are pre-zeroed.
 * V = e / n * (1 - r / sigma) ^ n
 * 
 * @param positions Pointer to the array of positions of the particles.
 * @param radii Pointer to the array of radii of the particles.
 * @param forces Pointer to the array of forces on the particles.
 * @param potential_energy Pointer to the array of potential energies of the particles.
 */
__global__ void kernelCalcDiskForces(const double* positions_x, const double* positions_y, const double* radii, double* forces_x, double* forces_y, double* potential_energy);

__global__ void kernelCalcDiskWallForces(const double* positions_x, const double* positions_y, const double* radii, double* forces_x, double* forces_y, double* potential_energy);

__global__ void kernelCalcDiskForceDistancePairs(const double* positions_x, const double* positions_y, double* potential_pairs, double* force_pairs_x, double* force_pairs_y, double* distance_pairs_x, double* distance_pairs_y, long* this_pair_id, long* other_pair_id, double* overlap_pairs, double* radsum_pairs, const double* radii, const long* static_particle_index, double* pair_separation_angle);

__global__ void kernelCountDiskContacts(const double* positions_x, const double* positions_y, const double* radii, long* contact_counts);

__global__ void kernelCalcDiskStressTensor(const double* positions_x, const double* positions_y, const double* velocities_x, const double* velocities_y, const double* masses, const double* radii, double* stress_tensor_x_x, double* stress_tensor_x_y, double* stress_tensor_y_x, double* stress_tensor_y_y);