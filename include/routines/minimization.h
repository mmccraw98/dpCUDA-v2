#pragma once

#include "../../include/particles/base/particle.h"
#include "../../include/integrator/adam.h"
#include "../../include/integrator/fire.h"
#include "../../include/io/io_manager.h"

void minimizeAdam(Particle& particle, double alpha, double beta1, double beta2, double epsilon, double avg_pe_target, double avg_pe_diff_target, long num_steps, long log_every_n);

void minimizeAdam(Particle& particle, double alpha, double beta1, double beta2, double epsilon, long num_steps, long log_every_n);

void minimizeAdam(Particle& particle);

void minimizeAdam(Particle& particle, double avg_pe_target, double avg_pe_diff_target);

void minimizeAdam(Particle& particle, long log_every_n);

void minimizeFire(Particle& particle, double alpha_init, double dt, double avg_pe_target, double avg_pe_diff_target, long num_steps, long log_every_n);

void minimizeFire(Particle& particle, double alpha_init, double dt, long num_steps, long log_every_n);

void minimizeFire(Particle& particle, long log_every_n);

void minimizeFire(Particle& particle, double avg_pe_target, double avg_pe_diff_target);

void minimizeFire(Particle& particle, double avg_pe_target, double avg_pe_diff_target, long num_steps);

void minimizeFire(Particle& particle);
