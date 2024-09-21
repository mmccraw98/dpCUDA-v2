#ifndef PARTICLE_FACTORY_H
#define PARTICLE_FACTORY_H

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "particle.h"
#include "disk.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class ParticleFactory {
public:
    virtual ~ParticleFactory() = default;

    /**
     * @brief Create a particle object.
     * 
     * @return A pointer to the created particle object.
     */
    virtual std::unique_ptr<Particle> createParticle() const = 0;
};

#endif /* PARTICLE_FACTORY_H */
