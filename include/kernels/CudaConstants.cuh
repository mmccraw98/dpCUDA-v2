// include/kernels/CudaConstants.cuh
#ifndef CUDA_CONSTANTS_CUH
#define CUDA_CONSTANTS_CUH

#include "Constants.h"
#include <cuda_runtime.h>  // This is needed for CUDA keywords like __constant__

// Constants are defined globally on the device - all threads can access them - they are updated using cudaMemcpy
// They should be small - only a few bytes

// Constants
__constant__ double d_box_size[N_DIM];  // Array of the box size (nDim, 1) vector

__constant__ long d_n_dim = N_DIM;  // Number of dimensions
__constant__ long d_n_particles;  // Number of particles
__constant__ long d_n_vertices;  // Number of vertices
__constant__ long d_n_dof;  // Number of degrees of freedom

#endif // CUDA_CONSTANTS_CUH