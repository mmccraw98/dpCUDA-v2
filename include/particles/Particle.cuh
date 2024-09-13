// include/particles/Particle.cuh
#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <any>
#include <typeinfo>
#include <unordered_map>

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
    long n_particles;
    long n_dim = 2;

    // --------------------- Utility Methods ---------------------

    std::unordered_map<std::string, std::any> getArrayMap();

    template <typename T>
    T getArrayValue(const std::string& array_name, size_t index);

    template <typename T>
    void setArrayValue(const std::string& array_name, size_t index, const T& value);

    // ------------------- Simulation Methods --------------------

    void updatePositions(double dt) {
        std::cout << "Particle::updatePositions" << std::endl;
        static_cast<Derived*>(this)->updatePositionsImpl(dt);
    }

    void updateMomenta(double dt) {
        std::cout << "Particle::updateMomenta" << std::endl;
        static_cast<Derived*>(this)->updateMomentaImpl(dt);
    }

    void calculateForces() {
        std::cout << "Particle::calculateForces" << std::endl;
        static_cast<Derived*>(this)->calculateForcesImpl();
    }

    double totalKineticEnergy() const {
        std::cout << "Particle::totalKineticEnergy" << std::endl;
        return static_cast<const Derived*>(this)->totalKineticEnergyImpl();
    }

    double totalPotentialEnergy() const {
        std::cout << "Particle::totalPotentialEnergy" << std::endl;
        return static_cast<const Derived*>(this)->totalPotentialEnergyImpl();
    }

    double totalEnergy() const {
        std::cout << "Particle::totalEnergy" << std::endl;
        return static_cast<const Derived*>(this)->totalEnergyImpl();
    }
};

#endif // PARTICLE_CUH