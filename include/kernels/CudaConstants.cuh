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

__constant__ long* d_numVertexInParticleListPtr;  // Pointer to the array that holds the number of vertices in each particle
__constant__ long* d_firstVertexIdInParticlePtr;  // Pointer to the array that holds the id of the first vertex in each particle
__constant__ long* d_particleIdListPtr;  // Pointer to the array that holds the particle id of each vertex

#endif // GLOBALS_CUH