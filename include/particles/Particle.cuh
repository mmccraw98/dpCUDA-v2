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

    std::unordered_map<std::string, std::any> getArrayMap() {
        std::unordered_map<std::string, std::any> array_map;

        // Double arrays
        array_map["d_positions"]        = &d_positions;
        array_map["d_momenta"]          = &d_momenta;
        array_map["d_forces"]           = &d_forces;
        array_map["d_radii"]            = &d_radii;
        array_map["d_masses"]           = &d_masses;
        array_map["d_potential_energy"] = &d_potential_energy;
        array_map["d_kinetic_energy"]   = &d_kinetic_energy;
        array_map["d_last_positions"]   = &d_last_positions;
        array_map["d_box_size"]         = &d_box_size;

        // Long arrays
        array_map["d_neighbor_list"]    = &d_neighbor_list;

        return array_map;
    }

    /**
     * @brief Retrieves the device array by name as a host vector.
     * 
     * @tparam T 
     * @param array_name name of the array to retrieve
     * @return thrust::host_vector<T> 
     */
    template <typename T>
    thrust::host_vector<T> getArray(const std::string& array_name) {
        auto array_map = getArrayMap();
        auto it = array_map.find(array_name);

        if (it != array_map.end()) {
            if (it->second.type() == typeid(thrust::device_vector<T>*)) {
                auto vec_ptr = std::any_cast<thrust::device_vector<T>*>(it->second);
                // Create a host vector and copy device data to host
                thrust::host_vector<T> host_array(vec_ptr->size());
                thrust::copy(vec_ptr->begin(), vec_ptr->end(), host_array.begin());
                return host_array;
            } else {
                throw std::runtime_error("Type mismatch for array: " + array_name);
            }
        } else {
            throw std::runtime_error("Array not found: " + array_name);
        }
    }

    template <typename T>
    void setArray(const std::string& array_name, const thrust::host_vector<T>& host_array) {
        auto array_map = getArrayMap();
        auto it = array_map.find(array_name);

        if (it != array_map.end()) {
            if (it->second.type() == typeid(thrust::device_vector<T>*)) {
                auto vec_ptr = std::any_cast<thrust::device_vector<T>*>(it->second);
                if (host_array.size() != vec_ptr->size()) {
                    throw std::out_of_range("Size mismatch between host and device arrays for: " + array_name);
                }
                // Copy host data back to device
                thrust::copy(host_array.begin(), host_array.end(), vec_ptr->begin());
            } else {
                throw std::runtime_error("Type mismatch for array: " + array_name);
            }
        } else {
            throw std::runtime_error("Array not found: " + array_name);
        }
    }
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