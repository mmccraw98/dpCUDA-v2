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

class Disk : public Particle {
public:
    // Constructor
    Disk(long n_particles, long seed = 0);

    // Destructor
    virtual ~Disk();

    // New device array
    thrust::device_vector<double> d_test_array;

    // Override getArrayMap to include d_test_array
    std::unordered_map<std::string, std::any> getArrayMap() {
        auto array_map = Particle::getArrayMap();
        array_map["d_test_array"] = &d_test_array;
        return array_map;
    }

    // Override pure virtual methods from Particle base class
    void initDynamicVariables() override;
    void clearDynamicVariables() override;
    void initGeometricVariables() override;
    void clearGeometricVariables() override;
    void setRandomPositions() override;

    double getArea() const override;
    double getOverlapFraction() const override;
    void scalePositions(double scale_factor) override;
    void updatePositions(double dt) override;
    void updateMomenta(double dt) override;
    void calculateForces() override;
    void calculateKineticEnergy() override;
    void updateNeighborList() override;
};

#endif /* DISK_H */
