#pragma once

#include "../../include/particles/base/particle.h"
#include "../../include/integrator/adam.h"
#include "../../include/io/io_manager.h"

void minimizeAdam(Particle& particle);