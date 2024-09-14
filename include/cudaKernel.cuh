#ifndef DPM2DKERNEL_CUH_
#define DPM2DKERNEL_CUH_

#include "defs.h"
#include <stdio.h>

__constant__ long d_dimBlock;
__constant__ long d_dimGrid;
__constant__ long d_partDimGrid;

__constant__ double* d_boxSizePtr;

__constant__ long d_nDim;
__constant__ long d_numParticles;
__constant__ long d_numVertexPerParticle;
__constant__ long d_numVertices;

__constant__ long* d_numVertexInParticleListPtr;
__constant__ long* d_firstVertexInParticleIdPtr;
__constant__ long* d_particleIdListPtr;
__constant__ double* d_vertexMassListPtr;

#endif /* DPM2DKERNEL_CUH_ */