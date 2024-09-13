
#ifndef CUDA_CONSTANTS_CUH_
#define CUDA_CONSTANTS_CUH_

#include "constants.h"
#include <stdio.h>

__constant__ long d_dim_block;
__constant__ long d_dim_grid;

__constant__ double d_box_size[N_DIM];

__constant__ long d_n_dim = N_DIM;
__constant__ long d_num_particles;
__constant__ long d_num_vertex_per_particle;
__constant__ long d_num_vertices;

#endif /* CUDA_CONSTANTS_CUH_ */