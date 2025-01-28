#pragma once

#include "../../include/data/data_2d.h"
#include "../../include/data/data_1d.h"
#include "../../include/particles/base/particle.h"
#include "../../include/integrator/adam.h"
#include "../../include/io/io_manager.h"
#include "../../include/particles/disk/disk.h"
#include "../../include/particles/disk/config.h"
#include "../../include/utils/config_dict.h"

std::tuple<SwapData2D<double>, SwapData1D<double>, SwapData1D<double>> get_minimal_overlap_disk_positions_and_radii(ConfigDict& config, double overcompression_factor = 0.0);