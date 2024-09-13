// include/kernels/VertexShapeInitializers.cuh
#ifndef VERTEXSHAPEINITIALIZERS_CUH
#define VERTEXSHAPEINITIALIZERS_CUH

#include "../Constants.h"
#include "CudaConstants.cuh"
#include "UtilityKernels.cuh"

__device__ void putVerticesOnCircles(double* vertex_positions, double* particle_positions, const double* circle_radii);

#endif // VERTEXSHAPEINITIALIZERS_CUH