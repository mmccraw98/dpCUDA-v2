// include/kernels/Globals.cuh
#ifndef GLOBALS_CUH
#define GLOBALS_CUH

#include "../Constants.h"

// Constants are defined globally on the device - all threads can access them - they are updated using cudaMemcpy

// Constants
__constant__ double* d_boxSizePointer;  // Pointer to the box size (nDim, 1) vector

__constant__ long d_nDim = N_DIM;  // Number of dimensions
__constant__ long d_numParticles;  // Number of particles
__constant__ long d_numVertices;  // Number of vertices

#endif // GLOBALS_CUH