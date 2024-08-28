#ifndef PARTICLE_H
#define PARTICLE_H

class Particle {
public:
    float mass;

    Particle(float mass);
    virtual ~Particle() = default;

    virtual void updatePosition(float dt) = 0;
    virtual void updateVelocity(float dt) = 0;
};

#endif // PARTICLE_H