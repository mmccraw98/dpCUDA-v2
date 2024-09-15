#ifndef DYNAMICS_KERNELS_CUH
#define DYNAMICS_KERNELS_CUH

#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "../../include/constants.h"
#include "../../include/cuda_constants.cuh"
#include "../../include/kernels/general.cuh"

__global__ void kernelPrintN();

/**
 * @brief Update the positions of the particles using an explicit Euler method.
 * Also updates the displacements of the particles from the last neighbor list update.
 * 
 * @param positions The positions of the particles.
 * @param last_positions The positions of the particles at the last time step.
 * @param displacements The displacements of the particles.
 * @param velocities The velocities of the particles.
 * @param dt The time step.
 */
__global__ void kernelUpdatePositions(double* positions, const double* last_positions, double* displacements, double* velocities, const double dt);

/**
 * @brief Update the velocities of the particles using an explicit Euler method.
 * 
 * @param velocities The velocities of the particles.
 * @param forces The forces on the particles.
 * @param masses The masses of the particles.
 * @param dt The time step.
 */
__global__ void kernelUpdateVelocities(double* velocities, double* forces, const double* masses, const double dt);

#endif // DYNAMICS_KERNELS_CUH