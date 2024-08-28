// include/particles/Particle.cuh
#ifndef PARTICLE_CUH
#define PARTICLE_CUH

class Particle {
public:
    virtual void update() = 0; // Pure virtual function
    virtual ~Particle() {}
};

#endif // PARTICLE_CUH