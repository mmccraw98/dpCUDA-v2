#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particles/base/particle.h"
#include "../../include/io/io_utils.h"
#include "../../include/particles/disk/disk.h"
#include "../../include/particles/disk/kernels.cuh"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

Disk::Disk() {
}

Disk::~Disk() {
}

// ----------------------------------------------------------------------
// --------------------- Overridden Methods -----------------------------
// ----------------------------------------------------------------------

long Disk::load(std::filesystem::path root_path, std::string source, long frame) {

    auto [frame_path, restart_path, init_path, frame_number] = this->getPaths(root_path, source, frame);

    ConfigDict config;
    config.load(root_path / "system" / "particle_config.json");

    long n_particles = config.at("n_particles").get<long>();
    long particle_dim_block = config.at("particle_dim_block").get<long>();
    double size_ratio = config.at("size_ratio").get<double>();
    double count_ratio = config.at("count_ratio").get<double>();
    double e_c = config.at("e_c").get<double>();
    double n_c = config.at("n_c").get<double>();
    double mass = config.at("mass").get<double>();
    double packing_fraction = config.at("packing_fraction").get<double>();
    long seed = config.at("seed").get<long>();

    // set the config
    this->setConfig(config);

    // set the seed
    this->setSeed(seed);

    // set the number of particles
    this->setNumParticles(n_particles);
    this->initDynamicVariables();
    // end

    // load the particle radii
    this->tryLoadData(frame_path, restart_path, init_path, source, "radii");
    // end

    this->setKernelDimensions(particle_dim_block);
    this->setDegreesOfFreedom();
    this->initGeometricVariables();

    // load/set the masses
    this->tryLoadData(frame_path, restart_path, init_path, source, "masses");
    // end

    // load/set the energy scales
    this->setEnergyScale(e_c, "c");
    this->setExponent(n_c, "c");
    // end

    // load the particle positions, velocities, and box size
    this->tryLoadData(frame_path, restart_path, init_path, source, "positions");
    this->tryLoadData(frame_path, restart_path, init_path, source, "velocities");
    this->tryLoadData(frame_path, restart_path, init_path, source, "box_size");
    this->calculateParticleArea();
    // calculate the packing fraction and set its value in the config
    config["packing_fraction"] = this->getPackingFraction();
    // end
    
    // neighbors
    this->setupNeighbors(config);

    // Define the unique dependencies
    this->define_unique_dependencies();

    return frame_number;
}

void Disk::initializeFromConfig(ConfigDict& config, bool minimize) {
    long n_particles = config.at("n_particles").get<long>();
    long particle_dim_block = config.at("particle_dim_block").get<long>();
    double size_ratio = config.at("size_ratio").get<double>();
    double count_ratio = config.at("count_ratio").get<double>();
    double e_c = config.at("e_c").get<double>();
    double n_c = config.at("n_c").get<double>();
    double mass = config.at("mass").get<double>();
    double packing_fraction = config.at("packing_fraction").get<double>();
    long seed = config.at("seed").get<long>();

    // set the config
    this->setConfig(config);
    // set the seed
    this->setSeed(seed);

    // set the number of particles
    this->setNumParticles(n_particles);
    this->initDynamicVariables();
    // end

    // load/set the particle radii
    this->setBiDispersity(size_ratio, count_ratio);
    // end

    this->setKernelDimensions(particle_dim_block);
    this->setDegreesOfFreedom();
    this->initGeometricVariables();

    // load/set the masses
    this->setMass(mass);
    // end

    // load/set the energy scales
    this->setEnergyScale(e_c, "c");
    this->setExponent(n_c, "c");
    // end

    // load/set the particle positions and angles
    this->initializeBox(packing_fraction);
    thrust::host_vector<double> h_box_size = this->getBoxSize();
    this->positions.fillRandomUniform(0.0, h_box_size[0], 0.0, h_box_size[1], 0.0, this->getSeed());
    this->calculateParticleArea();
    this->scaleToPackingFraction(packing_fraction);
    
    // neighbors
    this->setupNeighbors(config);

    // Minimize the positions here if desired
    if (minimize) {
        minimizeAdam(*this);
    }

    this->define_unique_dependencies();

    // this->setSeed(config.at("seed").get<long>());
    // this->setParticleCounts(config.at("n_particles").get<long>(), 0);
    // this->setKernelDimensions(config.at("particle_dim_block").get<long>());

    // this->setBiDispersity(config.at("size_ratio").get<double>(), config.at("count_ratio").get<double>());
    // this->initializeBox(config.at("packing_fraction").get<double>());

    // // TODO: make this a config - position initialization config: zero, random, etc.
    // this->setRandomPositions();

    // this->setEnergyScale(config.at("e_c").get<double>(), "c");
    // this->setExponent(config.at("n_c").get<double>(), "c");
    // this->setMass(config.at("mass").get<double>());

    // this->setupNeighbors(config);

    // this->setConfig(config);
}

void Disk::setKernelDimensions(long particle_dim_block) {
    int maxThreadsPerBlock;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    std::cout << "CUDA Info: Particle::setKernelDimensions: Max threads per block: " << maxThreadsPerBlock << std::endl;
    if (particle_dim_block > maxThreadsPerBlock) {
        std::cout << "WARNING: Particle::setKernelDimensions: particle_dim_block exceeds maxThreadsPerBlock, adjusting to maxThreadsPerBlock" << std::endl;
        particle_dim_block = maxThreadsPerBlock;
    }
    if (n_particles <= 0) {
        std::cout << "ERROR: Disk::setKernelDimensions: n_particles is 0.  Set n_particles before setting kernel dimensions." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (n_particles < particle_dim_block) {
        particle_dim_block = n_particles;
    }
    this->particle_dim_block = particle_dim_block;
    this->particle_dim_grid = (n_particles + particle_dim_block - 1) / particle_dim_block;

    if (n_vertices > 0) {
        std::cout << "WARNING: Disk::setKernelDimensions: n_vertices is " << n_vertices << ".  This is being ignored." << std::endl;
    }

    syncKernelDimensions();
}

// ----------------------------------------------------------------------
// ------------- Implementation of Pure Virtual Methods -----------------
// ----------------------------------------------------------------------


double Disk::getParticleArea() const {
    return thrust::transform_reduce(radii.d_vec.begin(), radii.d_vec.end(), Square(), 0.0, thrust::plus<double>()) * PI;
}

double Disk::getOverlapFraction() const {
    std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    // std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    // std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    // std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    // std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    return 0.0;
}

void Disk::calculateForces() {
    // kernelCalcDiskForces<<<particle_dim_grid, particle_dim_block>>>(d_positions_x_ptr, d_positions_y_ptr, d_radii_ptr, d_forces_x_ptr, d_forces_y_ptr, d_potential_energy_ptr);
    kernelCalcDiskForces<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, radii.d_ptr, forces.x.d_ptr, forces.y.d_ptr, potential_energy.d_ptr);
}

void Disk::calculateKineticEnergy() {
    // kernelCalculateTranslationalKineticEnergy<<<particle_dim_grid, particle_dim_block>>>(d_velocities_x_ptr, d_velocities_y_ptr, d_masses_ptr, d_kinetic_energy_ptr);
    kernelCalculateTranslationalKineticEnergy<<<particle_dim_grid, particle_dim_block>>>(velocities.x.d_ptr, velocities.y.d_ptr, masses.d_ptr, kinetic_energy.d_ptr);
}

void Disk::calculateForceDistancePairs() {
    potential_pairs.resizeAndFill(n_particles * max_neighbors_allocated, -1.0);
    force_pairs.resizeAndFill(n_particles * max_neighbors_allocated, 0.0, 0.0);
    distance_pairs.resizeAndFill(n_particles * max_neighbors_allocated, -1.0, -1.0);
    pair_ids.resizeAndFill(n_particles * max_neighbors_allocated, -1L, -1L);
    overlap_pairs.resizeAndFill(n_particles * max_neighbors_allocated, -1.0);
    radsum_pairs.resizeAndFill(n_particles * max_neighbors_allocated, -1.0);

    hessian_pairs_x.resizeAndFill(n_particles * max_neighbors_allocated, 0.0, 0.0);
    hessian_pairs_y.resizeAndFill(n_particles * max_neighbors_allocated, 0.0, 0.0);
    pair_separation_angle.resizeAndFill(n_particles * max_neighbors_allocated, -1.0);
    
    kernelCalcDiskForceDistancePairs<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, potential_pairs.d_ptr, force_pairs.x.d_ptr, force_pairs.y.d_ptr, distance_pairs.x.d_ptr, distance_pairs.y.d_ptr, pair_ids.x.d_ptr, pair_ids.y.d_ptr, overlap_pairs.d_ptr, radsum_pairs.d_ptr, radii.d_ptr, static_particle_index.d_ptr, pair_separation_angle.d_ptr, hessian_pairs_x.x.d_ptr, hessian_pairs_x.y.d_ptr, hessian_pairs_y.x.d_ptr, hessian_pairs_y.y.d_ptr);
}

void Disk::countContacts() {
    contact_counts.resizeAndFill(n_particles, 0L);
    kernelCountDiskContacts<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, radii.d_ptr, contact_counts.d_ptr
    );
}

void Disk::calculateWallForces() {
    kernelCalcDiskWallForces<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, radii.d_ptr, forces.x.d_ptr, forces.y.d_ptr, potential_energy.d_ptr);
}

void Disk::loadData(const std::string& root) {
    // unify all particle configs
    // add load functionality to configs


    // load config

    // set config

    // load data

    
    // SwapData2D<double> positions = read_2d_swap_data_from_file<double>(last_step_dir + "/positions.dat", particle_config.n_particles, 2);
    // Data1D<double> radii = read_1d_data_from_file<double>(source_path + "system/init/radii.dat", particle_config.n_particles);
}

void Disk::calculateStressTensor() {
    stress_tensor_x.resizeAndFill(n_particles, 0.0, 0.0);
    stress_tensor_y.resizeAndFill(n_particles, 0.0, 0.0);
    kernelCalcDiskStressTensor<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, velocities.x.d_ptr, velocities.y.d_ptr, masses.d_ptr, radii.d_ptr, stress_tensor_x.x.d_ptr, stress_tensor_x.y.d_ptr, stress_tensor_y.x.d_ptr, stress_tensor_y.y.d_ptr);
}