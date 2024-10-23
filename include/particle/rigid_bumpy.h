#pragma once

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "particle.h"
#include "config.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <nlohmann/json.hpp>


class RigidBumpy : public Particle {
public:
    RigidBumpy();
    virtual ~RigidBumpy();

    SwapData2D<double> vertex_positions;
    SwapData2D<double> vertex_velocities;
    SwapData2D<double> vertex_forces;

    // particle rotational variables
    SwapData1D<double> angles;
    SwapData1D<double> angular_velocities;
    SwapData1D<double> torques;

    // vertices should be all the same size and mass?

    void setKernelDimensions(long particle_dim_block = 256, long vertex_dim_block = 256);

    void initDynamicVariables();
    void clearDynamicVariables();

    double getArea() const override;

    double getOverlapFraction() const override;
    
    void calculateForces() override;

    void calculateKineticEnergy();
};