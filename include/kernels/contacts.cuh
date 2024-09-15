#ifndef CONTACTS_KERNELS_CUH
#define CONTACTS_KERNELS_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "../../include/constants.h"
#include "../../include/cuda_constants.cuh"
#include "../../include/kernels/general.cuh"

/**
 * @brief Update the neighbor list for all the particles.
 * 
 * @param positions The positions of the particles.
 * @param cutoff The cutoff distance for the neighbor list.
 */
__global__ void kernelUpdateNeighborList(const double* positions, const double cutoff);

#endif // CONTACTS_KERNELS_CUH