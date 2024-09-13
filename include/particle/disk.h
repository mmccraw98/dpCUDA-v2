#ifndef DISK_H
#define DISK_H

#include "../constants.h"
#include "particle.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>



class Disk : public Particle<Disk> {
public:
    // Constructor
    Disk(long n_particles, long seed=0);

    // Destructor
    ~Disk();

    // New device array
    thrust::device_vector<double> d_test_array;

    // Override getArrayMap to include d_test_array
    std::unordered_map<std::string, std::any> getArrayMap() {
        auto array_map = Particle<Disk>::getArrayMap();
        array_map["d_test_array"] = &d_test_array;
        return array_map;
    }

    /**
     * @brief Update particle positions
     * @param dt Time step
     */
    void updatePositionsImpl(double dt);

    void updateMomentaImpl(double dt);

    void calculateForcesImpl();

};

#endif /* DISK_H */