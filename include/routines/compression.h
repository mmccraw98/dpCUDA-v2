#pragma once

#include "../../include/particle/particle.h"
#include "../../include/integrator/adam.h"
#include "../../include/integrator/damped_nve.h"
#include "../../include/io/io_manager.h"

void jam_adam(Particle& particle, Adam& adam, IOManager& io_manager, long num_compression_steps, long num_adam_steps, double avg_pe_target, double avg_pe_diff_target, double packing_fraction_increment, double min_packing_fraction_increment, double max_pe_target);