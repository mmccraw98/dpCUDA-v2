
#ifndef CUDA_CONSTANTS_CUH_
#define CUDA_CONSTANTS_CUH_

#include "constants.h"
#include <stdio.h>

__constant__ long d_dim_block;
__constant__ long d_dim_grid;
__constant__ long d_dim_vertex_grid;

__constant__ double d_box_size[N_DIM];  // box size vector

__constant__ long d_n_dim = N_DIM;  // number of dimensions
__constant__ long d_n_particles;  // total number of particles
__constant__ long d_n_vertices;  // total number of vertices

#endif /* CUDA_CONSTANTS_CUH_ */