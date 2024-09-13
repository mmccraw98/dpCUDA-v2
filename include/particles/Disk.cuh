// include/particles/Disk.cuh
#ifndef DISK_CUH
#define DISK_CUH

#include "Particle.cuh"

class Disk : public Particle<Disk> {
public:
    // Constructor
    Disk(long N, long seed=0);

    // Destructor
    ~Disk();

    /**
     * @brief Update particle positions
     * @param dt Time step
     */
    void updatePositionsImpl(double dt);

    /**
     * @brief Update particle momenta
     * @param dt Time step
     */

    void updateMomentaImpl(double dt);

    void calculateForcesImpl();

};

#endif // DISK_CUH