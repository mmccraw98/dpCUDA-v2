#pragma once

#include "../utils/config_dict.h"
#include "../io/io_manager.h"
#include "../particles/base/particle.h"
#include "../integrator/nve.h"

void runNVTRescalingQuench(Particle& particle, double dt_dimless, long rescale_freq, double target_temperature, double temperature, double cooling_rate);