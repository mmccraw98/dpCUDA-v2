#pragma once

#include "../../include/data/data_2d.h"
#include "../../include/data/data_1d.h"
#include "../../include/particles/base/particle.h"
#include "../../include/integrator/adam.h"
#include "../../include/io/io_manager.h"
#include "../../include/particles/disk/disk.h"
#include "../../include/utils/config_dict.h"
#include "../../include/routines/minimization.h"


std::tuple<thrust::host_vector<double>, thrust::host_vector<double>, thrust::host_vector<double>, thrust::host_vector<double>> get_minimal_overlap_disk_positions_and_radii(ConfigDict& config, thrust::host_vector<double> positions_x, thrust::host_vector<double> positions_y, thrust::host_vector<double> radii, thrust::host_vector<double> box_size, double overcompression = 0.0);