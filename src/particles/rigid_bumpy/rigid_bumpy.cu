#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particles/base/particle.h"
#include "../../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../../include/particles/rigid_bumpy/kernels.cuh"
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
#include "../../include/particles/disk/disk.h"
#include "../../include/routines/initialization.h"


RigidBumpy::RigidBumpy() {
}

RigidBumpy::~RigidBumpy() {
}

// ----------------------------------------------------------------------
// --------------------- Overridden Methods -----------------------------
// ----------------------------------------------------------------------

void RigidBumpy::initializeFromConfig(ConfigDict& config) {
    auto [_positions, _radii, _box_size] = get_minimal_overlap_disk_positions_and_radii(config, 0.0);
    thrust::host_vector<double> h_radii = _radii.getData();

    // thrust::host_vector<long> vertex_particle_index_h;
    // vertex_particle_index_h = vertex_particle_index.getData();
    // std::cout << "0 1 vertex_particle_index_h[0]: " << vertex_particle_index_h[0] << std::endl;

    rotation = config["rotation"];
    segment_length_per_vertex_diameter = config["segment_length_per_vertex_diameter"];
    long n_vertices_per_small_particle = config["n_vertices_per_small_particle"];
    long particle_dim_block = config["particle_dim_block"];
    long vertex_dim_block = config["vertex_dim_block"];
    initializeVerticesFromDiskPacking(_positions, _radii, n_vertices_per_small_particle, particle_dim_block, vertex_dim_block);

    // vertex_particle_index_h = vertex_particle_index.getData();
    // std::cout << "0 2 vertex_particle_index_h[0]: " << vertex_particle_index_h[0] << std::endl;

    define_unique_dependencies();
    setSeed(config["seed"]);

    setRandomAngles();
    
    setBoxSize(_box_size.getData());

    // TODO: these have a bug
    // rb.calculateParticleArea();
    // rb.initializeBox(config.packing_fraction);

    config["vertex_radius"] = getVertexRadius();
    double geom_scale = getGeometryScale();
    double e_c = config["e_c"];
    config["e_c"] = e_c * (geom_scale * geom_scale);
    setEnergyScale(config["e_c"], "c");
    setExponent(config["n_c"], "c");
    setMass(config["mass"]);

    // setNeighborMethod(config["neighbor_list_config"]["neighbor_list_update_method"]);
    // setNeighborSize(config["neighbor_list_config"]["neighbor_cutoff_multiplier"], config["neighbor_list_config"]["neighbor_displacement_multiplier"]);
    // setCellSize(config["neighbor_list_config"]["num_particles_per_cell"], config["neighbor_list_config"]["cell_displacement_multiplier"]);
    // max_vertex_neighbors_allocated = 8;
    // syncVertexNeighborList();
    // initNeighborList();
    // syncVertexNeighborList();
    setupNeighbors(config);


    config["n_vertices"] = n_vertices;
    setConfig(config);

    // vertex_particle_index_h = vertex_particle_index.getData();
    // std::cout << "0 3 vertex_particle_index_h[0]: " << vertex_particle_index_h[0] << std::endl;
}

void RigidBumpy::finalizeLoading() {
    setParticleStartIndex();
    syncVertexIndices();
}

void RigidBumpy::loadDataFromPath(std::filesystem::path root_path, std::string data_file_extension) {
    Particle::loadDataFromPath(root_path, data_file_extension);
    std::cout << "RigidBumpy::loadDataFromPath: Loading data from " << root_path << " with extension " << data_file_extension << std::endl;
    for (const auto& entry : std::filesystem::directory_iterator(root_path)) {
        std::string filename = entry.path().filename().string();
        if (std::filesystem::is_directory(entry.path())) {
            continue;
        }
        filename = filename.substr(0, filename.find(data_file_extension));
        if (filename == "torques") {
            std::cout << "Loading torques" << std::endl;
            SwapData1D<double> loaded_torques = read_1d_swap_data_from_file<double>(entry.path().string(), n_particles);
            this->torques.setData(loaded_torques.getData());
        }
        else if (filename == "angles") {
            std::cout << "Loading angles" << std::endl;
            SwapData1D<double> loaded_angles = read_1d_swap_data_from_file<double>(entry.path().string(), n_particles);
            this->angles.setData(loaded_angles.getData());
        }
        else if (filename == "angular_velocities") {
            std::cout << "Loading angular_velocities" << std::endl;
            SwapData1D<double> loaded_angular_velocities = read_1d_swap_data_from_file<double>(entry.path().string(), n_particles);
            this->angular_velocities.setData(loaded_angular_velocities.getData());
        }
        else if (filename == "vertex_positions") {
            std::cout << "Loading vertex_positions" << std::endl;
            SwapData2D<double> loaded_vertex_positions = read_2d_swap_data_from_file<double>(entry.path().string(), n_vertices, 2);
            this->vertex_positions.setData(loaded_vertex_positions.getDataX(), loaded_vertex_positions.getDataY());
        }
        else if (filename == "vertex_velocities") {
            std::cout << "Loading vertex_velocities" << std::endl;
            SwapData2D<double> loaded_vertex_velocities = read_2d_swap_data_from_file<double>(entry.path().string(), n_vertices, 2);
            this->vertex_velocities.setData(loaded_vertex_velocities.getDataX(), loaded_vertex_velocities.getDataY());
        }
        else if (filename == "vertex_forces") {
            std::cout << "Loading vertex_forces" << std::endl;
            SwapData2D<double> loaded_vertex_forces = read_2d_swap_data_from_file<double>(entry.path().string(), n_vertices, 2);
            this->vertex_forces.setData(loaded_vertex_forces.getDataX(), loaded_vertex_forces.getDataY());
        }
        else if (filename == "static_vertex_index") {
            std::cout << "Loading static_vertex_index" << std::endl;
            Data1D<long> loaded_static_vertex_index = read_1d_data_from_file<long>(entry.path().string(), n_vertices);
            this->static_vertex_index.setData(loaded_static_vertex_index.getData());
        }
        else if (filename == "particle_start_index") {
            std::cout << "Loading particle_start_index" << std::endl;
            Data1D<long> loaded_particle_start_index = read_1d_data_from_file<long>(entry.path().string(), n_particles);
            this->particle_start_index.setData(loaded_particle_start_index.getData());
        }
        else if (filename == "vertex_particle_index") {
            std::cout << "Loading vertex_particle_index" << std::endl;
            Data1D<long> loaded_vertex_particle_index = read_1d_data_from_file<long>(entry.path().string(), n_vertices);
            this->vertex_particle_index.setData(loaded_vertex_particle_index.getData());
        }
        else if (filename == "num_vertices_in_particle") {
            std::cout << "Loading num_vertices_in_particle" << std::endl;
            Data1D<long> loaded_num_vertices_in_particle = read_1d_data_from_file<long>(entry.path().string(), n_particles);
            this->num_vertices_in_particle.setData(loaded_num_vertices_in_particle.getData());
        }
    }
}


void RigidBumpy::setKernelDimensions(long particle_dim_block, long vertex_dim_block) {
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

    if (vertex_dim_block > maxThreadsPerBlock) {
        std::cout << "WARNING: RigidBumpy::setKernelDimensions: vertex_dim_block exceeds maxThreadsPerBlock, adjusting to maxThreadsPerBlock" << std::endl;
        vertex_dim_block = maxThreadsPerBlock;
    }
    if (n_vertices <= 0) {
        std::cout << "ERROR: RigidBumpy::setKernelDimensions: n_vertices is 0.  Set n_vertices before setting kernel dimensions." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (n_vertices < vertex_dim_block) {
        vertex_dim_block = n_vertices;
    }
    this->vertex_dim_block = vertex_dim_block;
    this->vertex_dim_grid = (n_vertices + vertex_dim_block - 1) / vertex_dim_block;

    syncKernelDimensions();
}

void RigidBumpy::initVertexVariables() {
    vertex_positions.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_velocities.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_forces.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_torques.resizeAndFill(n_vertices, 0.0);
    vertex_particle_index.resizeAndFill(n_vertices, 0);
    vertex_masses.resizeAndFill(n_vertices, 0.0);
    vertex_potential_energy.resizeAndFill(n_vertices, 0.0);
    moments_of_inertia.resizeAndFill(n_particles, 0.0);
}

void RigidBumpy::initDynamicVariables() {
    Particle::initDynamicVariables();
    angles.resizeAndFill(n_particles, 0.0);
    angular_velocities.resizeAndFill(n_particles, 0.0);
    torques.resizeAndFill(n_particles, 0.0);
    angle_delta.resizeAndFill(n_particles, 0.0);
    delta.resizeAndFill(n_particles, 0.0, 0.0);
    particle_start_index.resizeAndFill(n_particles, 0);
    num_vertices_in_particle.resizeAndFill(n_particles, 0);
    inverse_particle_index.resizeAndFill(n_particles, 0);
    old_to_new_particle_index.resizeAndFill(n_particles, 0);
}

void RigidBumpy::initGeometricVariables() {
    area.resizeAndFill(n_particles, 0.0);
}

void RigidBumpy::clearDynamicVariables() {
    Particle::clearDynamicVariables();
    vertex_positions.clear();
    vertex_velocities.clear();
    vertex_forces.clear();
    vertex_torques.clear();
    vertex_potential_energy.clear();
    angles.clear();
    angular_velocities.clear();
    torques.clear();
    vertex_particle_index.clear();
    particle_start_index.clear();
    num_vertices_in_particle.clear();
    inverse_particle_index.clear();
    old_to_new_particle_index.clear();
    static_vertex_index.clear();
    vertex_index.clear();
}

void RigidBumpy::syncVertexIndices() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_particle_start_index_ptr, &particle_start_index.d_ptr, sizeof(particle_start_index.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexIndices: Error copying d_particle_start_index_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_num_vertices_in_particle_ptr, &num_vertices_in_particle.d_ptr, sizeof(num_vertices_in_particle.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexIndices: Error copying d_num_vertices_in_particle_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_vertex_particle_index_ptr, &vertex_particle_index.d_ptr, sizeof(vertex_particle_index.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexIndices: Error copying d_vertex_particle_index_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
}

void RigidBumpy::setParticleStartIndex() {
    thrust::exclusive_scan(num_vertices_in_particle.d_vec.begin(), num_vertices_in_particle.d_vec.end(), particle_start_index.d_vec.begin());
}

void RigidBumpy::syncVertexRadius(double vertex_radius) {
    cudaMemcpyToSymbol(d_vertex_radius, &vertex_radius, sizeof(double));
}

double RigidBumpy::getVertexRadius() {
    double vertex_radius;
    cudaMemcpyFromSymbol(&vertex_radius, d_vertex_radius, sizeof(double));
    return vertex_radius;
}

long RigidBumpy::setVertexBiDispersity(long num_vertices_in_small_particle) {
    double max_particle_diam = getDiameter("max");
    double min_particle_diam = getDiameter("min");

    auto num_small_particles = thrust::count_if(
        radii.d_vec.begin(), 
        radii.d_vec.end(), 
        [=] __device__ (double rad) {
            return rad == min_particle_diam / 2.0;
        }
    );

    auto num_large_particles = n_particles - num_small_particles;
    long num_vertices_in_large_particle = static_cast<long>(num_vertices_in_small_particle * max_particle_diam / min_particle_diam);

    double vertex_angle_small = 2 * M_PI / num_vertices_in_small_particle;
    double vertex_radius = min_particle_diam / (1 + segment_length_per_vertex_diameter / std::sin(vertex_angle_small / 2)) / 2.0;
    syncVertexRadius(vertex_radius);
    setNumVertices(num_small_particles * num_vertices_in_small_particle + num_large_particles * num_vertices_in_large_particle);
    return num_vertices_in_large_particle;
}

void RigidBumpy::setDegreesOfFreedom() {
    if (rotation) {
        this->n_dof = n_particles * (N_DIM + 1);  // two translation and one rotation
    } else {
        this->n_dof = n_particles * N_DIM;  // two translation
    }
}

void RigidBumpy::scaleVelocitiesToTemperature(double temperature) {
    double current_temp = calculateTemperature();
    if (current_temp <= 0.0) {
        std::cout << "WARNING: RigidBumpy::scaleVelocitiesToTemperature: Current temperature is " << current_temp << ", there will be an error!" << std::endl;
    }
    double scale_factor = std::sqrt(temperature / current_temp);
    velocities.scale(scale_factor, scale_factor);
    angular_velocities.scale(scale_factor);
}

void RigidBumpy::setRandomAngles() {
    angles.fillRandomUniform(0.0, 2 * M_PI, 0.0, seed);
}


void RigidBumpy::setRandomVelocities(double temperature) {
    velocities.fillRandomNormal(0.0, std::sqrt(temperature), 0.0, std::sqrt(temperature), 1, seed);
    angular_velocities.fillRandomNormal(0.0, std::sqrt(temperature), 1, seed);
    removeMeanVelocities();
    scaleVelocitiesToTemperature(temperature);
}

void RigidBumpy::initializeVerticesFromDiskPacking(SwapData2D<double>& disk_positions, SwapData1D<double>& disk_radii, long num_vertices_in_small_particle, long particle_dim_block, long vertex_dim_block) {
    
    
    thrust::host_vector<long> num_vertices_in_particle_h;
    thrust::host_vector<long> vertex_particle_index_h;
    thrust::host_vector<long> static_vertex_index_h;
    
    
    // set the number of particles from the disk data
    setNumParticles(disk_positions.size[0]);
    initDynamicVariables();
    initGeometricVariables();

    num_vertices_in_particle_h = num_vertices_in_particle.getData();
    std::cout << "1 num_vertices_in_particle_h[0]: " << num_vertices_in_particle_h[0] << std::endl;
    // vertex_particle_index_h = vertex_particle_index.getData();
    // std::cout << "1 vertex_particle_index_h[0]: " << vertex_particle_index_h[0] << std::endl;

    // set the particle positions and radii from the disk packing
    positions.copyFrom(disk_positions);
    radii.copyFrom(disk_radii);


    // define the number of vertices using the bidispersity
    long num_vertices_in_large_particle = setVertexBiDispersity(num_vertices_in_small_particle);

    num_vertices_in_particle_h = num_vertices_in_particle.getData();
    std::cout << "2 num_vertices_in_particle_h[0]: " << num_vertices_in_particle_h[0] << std::endl;
    // vertex_particle_index_h = vertex_particle_index.getData();
    // std::cout << "2 vertex_particle_index_h[0]: " << vertex_particle_index_h[0] << std::endl;

    setDegreesOfFreedom();

    // set the kernel dimensions
    setKernelDimensions(particle_dim_block, vertex_dim_block);

    // initialize the vertex variables
    initVertexVariables();

    num_vertices_in_particle_h = num_vertices_in_particle.getData();
    std::cout << "3 num_vertices_in_particle_h[0]: " << num_vertices_in_particle_h[0] << std::endl;
    vertex_particle_index_h = vertex_particle_index.getData();
    std::cout << "3 vertex_particle_index_h[0]: " << vertex_particle_index_h[0] << std::endl;
    // static_vertex_index_h = static_vertex_index.getData();
    // std::cout << "3 static_vertex_index_h[0]: " << static_vertex_index_h[0] << std::endl;

    double min_particle_diam = getDiameter("min");
    double max_particle_diam = getDiameter("max");


    // set the number of vertices in each particle
    kernelGetNumVerticesInParticles<<<particle_dim_grid, particle_dim_block>>>(
        radii.d_ptr, min_particle_diam, num_vertices_in_small_particle, max_particle_diam, num_vertices_in_large_particle, num_vertices_in_particle.d_ptr);

    num_vertices_in_particle_h = num_vertices_in_particle.getData();
    std::cout << "4 num_vertices_in_particle_h[0]: " << num_vertices_in_particle_h[0] << std::endl;
    vertex_particle_index_h = vertex_particle_index.getData();
    std::cout << "4 vertex_particle_index_h[0]: " << vertex_particle_index_h[0] << std::endl;
    // static_vertex_index_h = static_vertex_index.getData();
    // std::cout << "4 static_vertex_index_h[0]: " << static_vertex_index_h[0] << std::endl;

    // set the particle start index
    setParticleStartIndex();

    num_vertices_in_particle_h = num_vertices_in_particle.getData();
    std::cout << "5 num_vertices_in_particle_h[0]: " << num_vertices_in_particle_h[0] << std::endl;
    vertex_particle_index_h = vertex_particle_index.getData();
    std::cout << "5 vertex_particle_index_h[0]: " << vertex_particle_index_h[0] << std::endl;
    // static_vertex_index_h = static_vertex_index.getData();
    // std::cout << "5 static_vertex_index_h[0]: " << static_vertex_index_h[0] << std::endl;

    // set random angles
    angles.fillRandomUniform(0.0, 2 * M_PI, 0.0, seed);

    // initialize the vertices on the particles
    kernelInitializeVerticesOnParticles<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, radii.d_ptr, angles.d_ptr, vertex_particle_index.d_ptr, particle_start_index.d_ptr, num_vertices_in_particle.d_ptr, vertex_masses.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr);

    num_vertices_in_particle_h = num_vertices_in_particle.getData();
    std::cout << "6 num_vertices_in_particle_h[0]: " << num_vertices_in_particle_h[0] << std::endl;
    vertex_particle_index_h = vertex_particle_index.getData();
    std::cout << "6 vertex_particle_index_h[0]: " << vertex_particle_index_h[0] << std::endl;
    // static_vertex_index_h = static_vertex_index.getData();
    // std::cout << "6 static_vertex_index_h[0]: " << static_vertex_index_h[0] << std::endl;

    // sync the vertex indices
    syncVertexIndices();

    num_vertices_in_particle_h = num_vertices_in_particle.getData();
    std::cout << "7 num_vertices_in_particle_h[0]: " << num_vertices_in_particle_h[0] << std::endl;
    vertex_particle_index_h = vertex_particle_index.getData();
    std::cout << "7 vertex_particle_index_h[0]: " << vertex_particle_index_h[0] << std::endl;
    // static_vertex_index_h = static_vertex_index.getData();
    // std::cout << "7 static_vertex_index_h[0]: " << static_vertex_index_h[0] << std::endl;
}

void RigidBumpy::loadData(const std::string& root) {
    // TODO: implement this
}

ArrayData RigidBumpy::getArrayData(const std::string& array_name) {
    try {
        return Particle::getArrayData(array_name);
    } catch (std::invalid_argument& e) {
        // try the rigid bumpy specific ones
        ArrayData result;
        result.name = array_name;
        if (array_name == "vertex_positions") {
            result.type = DataType::Double;
            result.size = vertex_positions.size;
            result.data = std::make_pair(vertex_positions.getDataX(), vertex_positions.getDataY());
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "vertex_velocities") {
            result.type = DataType::Double;
            result.size = vertex_velocities.size;
            result.data = std::make_pair(vertex_velocities.getDataX(), vertex_velocities.getDataY());
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "vertex_forces") {
            result.type = DataType::Double;
            result.size = vertex_forces.size;
            result.data = std::make_pair(vertex_forces.getDataX(), vertex_forces.getDataY());
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "vertex_masses") {
            result.type = DataType::Double;
            result.size = vertex_masses.size;
            result.data = vertex_masses.getData();
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "angles") {
            result.type = DataType::Double;
            result.size = angles.size;
            result.data = angles.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "angular_velocities") {
            result.type = DataType::Double;
            result.size = angular_velocities.size;
            result.data = angular_velocities.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "torques") {
            result.type = DataType::Double;
            result.size = torques.size;
            result.data = torques.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "area") {
            result.type = DataType::Double;
            result.size = area.size;
            result.data = area.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "vertex_particle_index") {
            result.type = DataType::Long;
            result.size = vertex_particle_index.size;
            result.data = vertex_particle_index.getData();
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "particle_start_index") {
            result.type = DataType::Long;
            result.size = particle_start_index.size;
            result.data = particle_start_index.getData();
            result.index_array_name = "";
        } else if (array_name == "num_vertices_in_particle") {
            result.type = DataType::Long;
            result.size = num_vertices_in_particle.size;
            result.data = num_vertices_in_particle.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "vertex_neighbor_list") {
            result.type = DataType::Long;
            result.size = vertex_neighbor_list.size;
            result.data = vertex_neighbor_list.getData();
            result.index_array_name = "";// TODO:
        } else if (array_name == "num_vertex_neighbors") {
            result.type = DataType::Long;
            result.size = num_vertex_neighbors.size;
            result.data = num_vertex_neighbors.getData();
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "vertex_index") {
            result.type = DataType::Long;
            result.size = vertex_index.size;
            result.data = vertex_index.getData();
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "static_vertex_index") {
            result.type = DataType::Long;
            result.size = static_vertex_index.size;
            result.data = static_vertex_index.getData();
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "moments_of_inertia") {
            result.type = DataType::Double;
            result.size = moments_of_inertia.size;
            result.data = moments_of_inertia.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "vertex_torques") {
            result.type = DataType::Double;
            result.size = vertex_torques.size;
            result.data = vertex_torques.getData();
            result.index_array_name = "static_vertex_index";

        } else {
            throw std::invalid_argument("RigidBumpy::getArrayData: array_name " + array_name + " not found");
        }
        return result;
    }
}





// need to make a scale function for the particles which can then go into the base particle class and be overridden by the rigid bumpy class so we dont have to replicate the scaleToPackingFraction function

void RigidBumpy::calculateParticleArea() {
    // kernelCalculateParticlePolygonArea<<<particle_dim_grid, particle_dim_block>>>(
    //     vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, area.d_ptr);
    kernelCalculateBumpyParticleAreaFull<<<particle_dim_grid, particle_dim_block>>>(
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, radii.d_ptr, area.d_ptr);
}

double RigidBumpy::getParticleArea() const {
    double a = thrust::reduce(area.d_vec.begin(), area.d_vec.end(), 0.0, thrust::plus<double>());
    double box_area = getBoxArea();
    return a;
}


// calculate the particle positions first if not already done not needed for rigid bumpy
// compare polygon areas to particle areas
// calculate the contribution to the area from the vertices
void RigidBumpy::scalePositions(double scale_factor) {
    kernelScalePositions<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, scale_factor
    );
}

void RigidBumpy::syncVertexNeighborList() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_max_vertex_neighbors_allocated, &max_vertex_neighbors_allocated, sizeof(max_vertex_neighbors_allocated));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexNeighborList: Error copying max_vertex_neighbors_allocated to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_vertex_neighbor_list_ptr, &vertex_neighbor_list.d_ptr, sizeof(vertex_neighbor_list.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexNeighborList: Error copying d_vertex_neighbor_list_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_num_vertex_neighbors_ptr, &num_vertex_neighbors.d_ptr, sizeof(num_vertex_neighbors.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexNeighborList: Error copying d_num_vertex_neighbors_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
}

void RigidBumpy::setMass(double mass) {
    Particle::setMass(mass);
    kernelGetVertexMasses<<<particle_dim_grid, particle_dim_block>>>(
        radii.d_ptr, vertex_masses.d_ptr, masses.d_ptr);

    kernelGetMomentsOfInertia<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, vertex_masses.d_ptr, moments_of_inertia.d_ptr);

    // check if sum of vertex masses is equal to particle mass for each particle
    double total_vertex_mass = thrust::reduce(vertex_masses.d_vec.begin(), vertex_masses.d_vec.end(), 0.0, thrust::plus<double>());
    if (std::abs(total_vertex_mass / n_particles - mass) > 1e-14) {
        std::cout << "WARNING: RigidBumpy::setMass: Total vertex mass does not match particle mass" << std::endl;
    }
}

double RigidBumpy::getOverlapFraction() const {
    std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    return 0.0;
}

void RigidBumpy::calculateForces() {
    // forces.fill(0.0, 0.0);
    // torques.fill(0.0);
    // vertex_torques.fill(0.0);
    // vertex_forces.fill(0.0, 0.0);
    // vertex_potential_energy.fill(0.0);
    // potential_energy.fill(0.0);
    // version 1: 2 kernels, 1 vertex level and 1 particle level
    kernelCalcRigidBumpyForces1<<<vertex_dim_grid, vertex_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, vertex_forces.x.d_ptr, vertex_forces.y.d_ptr, vertex_torques.d_ptr, vertex_potential_energy.d_ptr);
    kernelCalcRigidBumpyParticleForces1<<<particle_dim_grid, particle_dim_block>>>(
        vertex_forces.x.d_ptr, vertex_forces.y.d_ptr, vertex_torques.d_ptr, vertex_potential_energy.d_ptr, forces.x.d_ptr, forces.y.d_ptr, torques.d_ptr, potential_energy.d_ptr);

    // // version 2: 1 particle level kernel  -- 3 times slower than version 1
    // kernelCalcRigidBumpyForces2<<<particle_dim_grid, particle_dim_block>>>(
    //     positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, forces.x.d_ptr, forces.y.d_ptr, torques.d_ptr, potential_energy.d_ptr
    // );
}

void RigidBumpy::updatePositions(double dt) {
    kernelUpdateRigidPositions<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, angles.d_ptr, delta.x.d_ptr, delta.y.d_ptr, angle_delta.d_ptr, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, last_cell_positions.x.d_ptr, last_cell_positions.y.d_ptr, neigh_displacements_sq.d_ptr, cell_displacements_sq.d_ptr, velocities.x.d_ptr, velocities.y.d_ptr, angular_velocities.d_ptr, dt
    );
    
    // version 1: vertex level
    kernelTranslateAndRotateVertices1<<<vertex_dim_grid, vertex_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, delta.x.d_ptr, delta.y.d_ptr, angle_delta.d_ptr);

    // // version 2: particle level  -- 10% slower than version 1
    // kernelTranslateAndRotateVertices2<<<particle_dim_grid, particle_dim_block>>>(
    //     positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, delta.x.d_ptr, delta.y.d_ptr, angle_delta.d_ptr
    // );
}

void RigidBumpy::updateVelocities(double dt) {
    kernelUpdateRigidVelocities<<<particle_dim_grid, particle_dim_block>>>(
        velocities.x.d_ptr, velocities.y.d_ptr, angular_velocities.d_ptr, forces.x.d_ptr, forces.y.d_ptr, torques.d_ptr, masses.d_ptr, moments_of_inertia.d_ptr, dt, rotation
    );
}

void RigidBumpy::calculateKineticEnergy() {
    if (rotation) {
        kernelCalculateTranslationalAndRotationalKineticEnergy<<<particle_dim_grid, particle_dim_block>>>(velocities.x.d_ptr, velocities.y.d_ptr, masses.d_ptr, angular_velocities.d_ptr, moments_of_inertia.d_ptr, kinetic_energy.d_ptr);
    } else {
        kernelCalculateTranslationalKineticEnergy<<<particle_dim_grid, particle_dim_block>>>(velocities.x.d_ptr, velocities.y.d_ptr, masses.d_ptr, kinetic_energy.d_ptr);
    }
}

void RigidBumpy::calculateParticlePositions() {
    kernelCalculateParticlePositions<<<particle_dim_grid, particle_dim_block>>>(
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, positions.x.d_ptr, positions.y.d_ptr
    );
}

void RigidBumpy::updateVertexVerletList() {
    vertex_neighbor_list.fill(-1L);
    kernelUpdateVertexNeighborList<<<vertex_dim_grid, vertex_dim_block>>>(
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, positions.x.d_ptr, positions.y.d_ptr, vertex_neighbor_cutoff, vertex_particle_neighbor_cutoff);
    long max_vertex_neighbors = thrust::reduce(num_vertex_neighbors.d_vec.begin(), num_vertex_neighbors.d_vec.end(), -1L, thrust::maximum<long>());
    if (max_vertex_neighbors > max_vertex_neighbors_allocated) {
        max_vertex_neighbors_allocated = std::pow(2, std::ceil(std::log2(max_vertex_neighbors)));
        std::cout << "RigidBumpy::updateVertexVerletList: Resizing neighbor list to " << max_vertex_neighbors_allocated << std::endl;
        vertex_neighbor_list.resizeAndFill(n_vertices * max_vertex_neighbors_allocated, -1L);
        syncVertexNeighborList();
        kernelUpdateVertexNeighborList<<<vertex_dim_grid, vertex_dim_block>>>(
            vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, positions.x.d_ptr, positions.y.d_ptr, vertex_neighbor_cutoff, vertex_particle_neighbor_cutoff);
    }
}

void RigidBumpy::updateVerletList() {
    Particle::updateVerletList();
    updateVertexVerletList();
}

void RigidBumpy::initVerletListVariables() {
    Particle::initVerletListVariables();
    vertex_neighbor_list.resizeAndFill(n_vertices * max_vertex_neighbors_allocated, -1L);
    num_vertex_neighbors.resizeAndFill(n_vertices, 0L);
    syncVertexNeighborList();
}

void RigidBumpy::initVerletList() {
    initVerletListVariables();
    syncNeighborList();
    syncVertexNeighborList();
    updateVerletList();
}

void RigidBumpy::initCellList() {
    initVerletListVariables();
    syncNeighborList();
    syncVertexNeighborList();
    initCellListVariables();
    updateCellList();
    updateCellNeighborList();
}

void RigidBumpy::updateCellNeighborList() {
    Particle::updateCellNeighborList();
    // TODO: i think we need to make vertex ids and other variables global in the kernels?
    updateVertexVerletList();
}

void RigidBumpy::initCellListVariables() {
    Particle::initCellListVariables();
    vertex_index.resize(n_vertices);
    static_vertex_index.resize(n_vertices);
    thrust::sequence(vertex_index.d_vec.begin(), vertex_index.d_vec.end());
    thrust::sequence(static_vertex_index.d_vec.begin(), static_vertex_index.d_vec.end());
}

double RigidBumpy::getGeometryScale() {
    double vertex_diameter = 2.0 * getVertexRadius();
    double particle_diameter = getDiameter("max");
    return vertex_diameter / particle_diameter;
}


bool RigidBumpy::setNeighborSize(double neighbor_cutoff_multiplier, double neighbor_displacement_multiplier) {
    this->max_neighbors_allocated = 4;  // initial assumption, probably could be refined
    this->max_vertex_neighbors_allocated = 4;  // initial assumption, probably could be refined
    this->neighbor_cutoff = neighbor_cutoff_multiplier * getDiameter("max");
    this->vertex_neighbor_cutoff = neighbor_cutoff_multiplier * 2.0 * getVertexRadius();
    this->vertex_particle_neighbor_cutoff = getDiameter("max");  // particles within this distance of a vertex will be checked for vertex neighbors
    this->neighbor_displacement_threshold_sq = std::pow(neighbor_displacement_multiplier * vertex_neighbor_cutoff, 2);
    thrust::host_vector<double> host_box_size = box_size.getData();
    double box_diagonal = std::sqrt(host_box_size[0] * host_box_size[0] + host_box_size[1] * host_box_size[1]);
    if (neighbor_cutoff >= box_diagonal) {
        std::cout << "Particle::setNeighborSize: Neighbor radius exceeds the box size" << std::endl;
        return false;
    }
    return true;

    // rb.vertex_neighbor_cutoff = 2.0 * vertex_diameter;  // vertices within this distance of each other are neighbors
}


void RigidBumpy::reorderParticleData() {
    // Sort particles by cell index
    thrust::sort_by_key(cell_index.d_vec.begin(), cell_index.d_vec.end(), 
        thrust::make_zip_iterator(thrust::make_tuple(
            particle_index.d_vec.begin(), 
            static_particle_index.d_vec.begin()
        ))
    );

    // Reorder particle data and create inverse mapping
    kernelReorderRigidBumpyParticleData<<<particle_dim_grid, particle_dim_block>>>(
        particle_index.d_ptr,
        old_to_new_particle_index.d_ptr,
        num_vertices_in_particle.d_ptr,
        num_vertices_in_particle.d_temp_ptr,
        positions.x.d_ptr, positions.y.d_ptr,
        positions.x.d_temp_ptr, positions.y.d_temp_ptr,
        forces.x.d_ptr, forces.y.d_ptr,
        forces.x.d_temp_ptr, forces.y.d_temp_ptr,
        velocities.x.d_ptr, velocities.y.d_ptr,
        velocities.x.d_temp_ptr, velocities.y.d_temp_ptr,
        angular_velocities.d_ptr, torques.d_ptr,
        angular_velocities.d_temp_ptr, torques.d_temp_ptr,
        masses.d_ptr, radii.d_ptr,
        masses.d_temp_ptr, radii.d_temp_ptr,
        moments_of_inertia.d_ptr,
        moments_of_inertia.d_temp_ptr,
        last_cell_positions.x.d_ptr, last_cell_positions.y.d_ptr,
        cell_displacements_sq.d_ptr
    );

    // Calculate new particle start indices using the temp data
    thrust::exclusive_scan(
        num_vertices_in_particle.d_temp_vec.begin(),
        num_vertices_in_particle.d_temp_vec.end(),
        particle_start_index.d_temp_vec.begin()
    );

    // Reorder vertex data
    kernelReorderRigidBumpyVertexData<<<vertex_dim_grid, vertex_dim_block>>>(
        vertex_particle_index.d_ptr,
        vertex_particle_index.d_temp_ptr,
        old_to_new_particle_index.d_ptr,
        particle_start_index.d_ptr,
        particle_start_index.d_temp_ptr,
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr,
        vertex_positions.x.d_temp_ptr, vertex_positions.y.d_temp_ptr,
        static_vertex_index.d_ptr,
        static_vertex_index.d_temp_ptr
    );

    // Swap all reordered data
    positions.swap();
    forces.swap();
    velocities.swap();
    angular_velocities.swap();
    torques.swap();
    masses.swap();
    radii.swap();
    moments_of_inertia.swap();
    num_vertices_in_particle.swap();
    particle_start_index.swap();
    vertex_positions.swap();
    vertex_particle_index.swap();
    first_moment.swap();
    second_moment.swap();
    static_vertex_index.swap();
    syncVertexIndices();
}

void RigidBumpy::zeroForceAndPotentialEnergy() {
    kernelZeroRigidBumpyParticleForceAndPotentialEnergy<<<particle_dim_grid, particle_dim_block>>>(forces.x.d_ptr, forces.y.d_ptr, torques.d_ptr, potential_energy.d_ptr);
    kernelZeroRigidBumpyVertexForceAndPotentialEnergy<<<vertex_dim_grid, vertex_dim_block>>>(vertex_forces.x.d_ptr, vertex_forces.y.d_ptr, vertex_torques.d_ptr, vertex_potential_energy.d_ptr);
}

void RigidBumpy::calculateForceDistancePairs() {
    force_pairs.resizeAndFill(n_particles * max_neighbors_allocated, 0.0, 0.0);
    distance_pairs.resizeAndFill(n_particles * max_neighbors_allocated, -1.0, -1.0);
    pair_ids.resizeAndFill(n_particles * max_neighbors_allocated, -1L, -1L);
    overlap_pairs.resizeAndFill(n_particles * max_neighbors_allocated, -1.0);
    radsum_pairs.resizeAndFill(n_particles * max_neighbors_allocated, -1.0);

    pos_pairs_i.resizeAndFill(n_particles * max_neighbors_allocated, -1.0, -1.0);
    pos_pairs_j.resizeAndFill(n_particles * max_neighbors_allocated, -1.0, -1.0);

    kernelCalcRigidBumpyForceDistancePairs<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, force_pairs.x.d_ptr, force_pairs.y.d_ptr, distance_pairs.x.d_ptr, distance_pairs.y.d_ptr, pair_ids.x.d_ptr, pair_ids.y.d_ptr, overlap_pairs.d_ptr, radsum_pairs.d_ptr, radii.d_ptr, static_particle_index.d_ptr, pos_pairs_i.x.d_ptr, pos_pairs_i.y.d_ptr, pos_pairs_j.x.d_ptr, pos_pairs_j.y.d_ptr); 


// __global__ void kernelCalcRigidBumpyForceDistancePairs(const double* positions_x, const double* positions_y, const double* vertex_positions_x, const double* vertex_positions_y, double* force_pairs_x, double* force_pairs_y, double* distance_pairs_x, double* distance_pairs_y, long* this_pair_id, long* other_pair_id, double* overlap_pairs, double* radsum_pairs, const double* radii, const long* static_particle_index, double* pos_pairs_i_x, double* pos_pairs_i_y, double* pos_pairs_j_x, double* pos_pairs_j_y) {

}

void RigidBumpy::initAdamVariables() {
    Particle::initAdamVariables();
    first_moment_angle.resizeAndFill(n_particles, 0.0);
    second_moment_angle.resizeAndFill(n_particles, 0.0);
}

void RigidBumpy::clearAdamVariables() {
    Particle::clearAdamVariables();
    first_moment_angle.clear();
    second_moment_angle.clear();
}

void RigidBumpy::updatePositionsAdam(long step, double alpha, double beta1, double beta2, double epsilon) {
    double one_minus_beta1_pow_t = 1 - pow(beta1, step + 1);
    double one_minus_beta2_pow_t = 1 - pow(beta2, step + 1);
    kernelRigidBumpyAdamStep<<<particle_dim_grid, particle_dim_block>>>(
        first_moment.x.d_ptr, first_moment.y.d_ptr, first_moment_angle.d_ptr, second_moment.x.d_ptr, second_moment.y.d_ptr, second_moment_angle.d_ptr, positions.x.d_ptr, positions.y.d_ptr, angles.d_ptr, delta.x.d_ptr, delta.y.d_ptr, angle_delta.d_ptr, forces.x.d_ptr, forces.y.d_ptr, torques.d_ptr, alpha, beta1, beta2, one_minus_beta1_pow_t, one_minus_beta2_pow_t, epsilon, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, neigh_displacements_sq.d_ptr, last_cell_positions.x.d_ptr, last_cell_positions.y.d_ptr, cell_displacements_sq.d_ptr, rotation);
    // kernelUpdateRigidPositions<<<particle_dim_grid, particle_dim_block>>>(
    //     positions.x.d_ptr, positions.y.d_ptr, angles.d_ptr, delta.x.d_ptr, delta.y.d_ptr, angle_delta.d_ptr, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, last_cell_positions.x.d_ptr, last_cell_positions.y.d_ptr, neigh_displacements_sq.d_ptr, cell_displacements_sq.d_ptr, velocities.x.d_ptr, velocities.y.d_ptr, angular_velocities.d_ptr, dt
    // );

    kernelTranslateAndRotateVertices1<<<vertex_dim_grid, vertex_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, delta.x.d_ptr, delta.y.d_ptr, angle_delta.d_ptr);
}

void RigidBumpy::calculateWallForces() {
    // calculates the forces from the walls on the vertices which is then later applied to the total particle forces
    kernelCalcRigidBumpyWallForces<<<vertex_dim_grid, vertex_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, vertex_forces.x.d_ptr, vertex_forces.y.d_ptr, vertex_torques.d_ptr, vertex_potential_energy.d_ptr
    );
}

void RigidBumpy::calculateDampedForces(double damping_coefficient) {
    kernelCalculateRigidDampedForces<<<particle_dim_grid, particle_dim_block>>>(forces.x.d_ptr, forces.y.d_ptr, torques.d_ptr, velocities.x.d_ptr, velocities.y.d_ptr, angular_velocities.d_ptr, damping_coefficient);
}