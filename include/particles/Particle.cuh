// include/particles/Particle.cuh
#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename Derived>
class Particle {
public:

    thrust::device_vector<double> d_positions;
    thrust::device_vector<double> d_momenta;
    thrust::device_vector<double> d_forces;
    thrust::device_vector<double> d_radii;
    thrust::device_vector<double> d_masses;
    thrust::device_vector<double> d_potential_energy;
    thrust::device_vector<double> d_kinetic_energy;
    thrust::device_vector<double> d_last_positions;
    thrust::device_vector<long> d_neighbor_list;
    thrust::device_vector<double> d_box_size;
    double e_c;

    void updatePositions(double dt) {
        static_cast<Derived*>(this)->updatePositionsImpl(dt);
    }

    void updateMomenta(double dt) {
        static_cast<Derived*>(this)->updateMomentaImpl(dt);
    }

    void calculateForces() {
        static_cast<Derived*>(this)->calculateForcesImpl();
    }

    double totalKineticEnergy() const {
        return static_cast<const Derived*>(this)->totalKineticEnergyImpl();
    }

    double totalPotentialEnergy() const {
        return static_cast<const Derived*>(this)->totalPotentialEnergyImpl();
    }

    double totalEnergy() const {
        return static_cast<const Derived*>(this)->totalEnergyImpl();
    }
};

#endif // PARTICLE_CUH