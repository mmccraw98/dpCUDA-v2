#ifndef ELLIPSE_H
#define ELLIPSE_H

#include "../constants.h"
#include "particle.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class Ellipse : public Particle {
public:
    Ellipse();

    virtual ~Ellipse();

    thrust::device_vector<double> d_test_array;  // example new device array to be added to the array map

    /**
     * @brief Get a key-value map to pointers for all the member device arrays.
     * Serves as a utility function for the getArray and setArray methods to reduce
     * code duplication.
     * 
     * @return The array map.
     */
    std::unordered_map<std::string, std::any> getArrayMap() {
        auto array_map = Particle::getArrayMap();
        array_map["d_test_array"] = &d_test_array;
        return array_map;
    }

    // ----------------------------------------------------------------------
    // ------------- Implementation of Pure Virtual Methods -----------------
    // ----------------------------------------------------------------------

    /**
     * @brief Initialize the dynamic variables.
     */
    void initDynamicVariables() override;

    /**
     * @brief Clear the dynamic variables.
     */
    void clearDynamicVariables() override;
    
    /**
     * @brief Empty implementation of the initGeometricVariables method.
     */
    void initGeometricVariables() override;

    /**
     * @brief Empty implementation of the clearGeometricVariables method.
     */
    void clearGeometricVariables() override;

    /**
     * @brief Get the total area of the particles by summing the squares of the radii.
     * 
     * @return The total area of the particles.
     */
    double getArea() const override;

    /**
     * @brief Get the fraction of the area involving the overlap between particles using the lense formula.
     * 
     * @return The overlap fraction of the particles.
     */
    double getOverlapFraction() const override;

    /**
     * @brief Apply an affine transformation to the positions of the particles.
     * 
     * @param scale_factor The scaling factor.
     */
    void scalePositions(double scale_factor) override;

    /**
     * @brief Update the positions of the particles using an explicit Euler method.
     * 
     * @param dt The time step.
     */
    void updatePositions(double dt) override;

    /**
     * @brief Update the momenta of the particles using an explicit Euler method.
     * 
     * @param dt The time step.
     */
    void updateMomenta(double dt) override;
    
    /**
     * @brief Calculate the forces on the particles.
     */
    void calculateForces() override;

    /**
     * @brief Calculate the kinetic energy of the particles.
     */
    void calculateKineticEnergy() override;
    
    /**
     * @brief Update the neighbor list of the particles.
     */
    void updateNeighborList() override;
};

#endif /* ELLIPSE_H */
