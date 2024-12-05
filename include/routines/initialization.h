#pragma once

#include "../../include/data/data_2d.h"
#include "../../include/data/data_1d.h"
#include "../../include/particle/config.h"
#include "../../include/particle/particle.h"
#include "../../include/integrator/adam.h"
#include "../../include/io/io_manager.h"
#include "../../include/particle/particle_factory.h"

std::tuple<SwapData2D<double>, SwapData1D<double>, SwapData1D<double>> get_minimal_overlap_positions_and_radii(BaseParticleConfig& config, double overcompression_factor = 0.0);
