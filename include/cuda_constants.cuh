// KERNEL FUNCTIONS THAT ACT ON THE DEVICE(GPU)

#ifndef CUDA_CONSTANTS_CUH_
#define CUDA_CONSTANTS_CUH_

#include "constants.h"
#include <stdio.h>

__constant__ long d_dimBlock;
__constant__ long d_dimGrid;
__constant__ long d_partDimGrid;

__constant__ double d_box_size[N_DIM];

__constant__ double* d_boxSizePtr;

__constant__ long d_nDim = N_DIM;
__constant__ long d_numParticles;
__constant__ long d_numVertexPerParticle;
__constant__ long d_numVertices;

#endif /* CUDA_CONSTANTS_CUH_ */