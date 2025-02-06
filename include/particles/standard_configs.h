#pragma once

#include "../include/utils/config_dict.h"

#include <string>

inline ConfigDict get_standard_disk_config(long n_particles, double packing_fraction) {
    ConfigDict config;
    config["particle_type"] = "disk";
    config["n_particles"] = n_particles;
    config["packing_fraction"] = packing_fraction;
    config["particle_dim_block"] = 256;
    config["e_c"] = 1.0;
    config["n_c"] = 2.0;
    config["mass"] = 1.0;
    config["size_ratio"] = 1.4;
    config["count_ratio"] = 0.5;
    config["seed"] = -1;
    config["neighbor_list_config"] = {
        {"neighbor_cutoff_multiplier", 1.5},
        {"neighbor_displacement_multiplier", 0.2},
        {"num_particles_per_cell", 8.0},
        {"cell_displacement_multiplier", 0.5},
        {"neighbor_list_update_method", "cell"}
    };
    return config;
}

inline ConfigDict get_standard_rigid_bumpy_config(long n_particles, double packing_fraction) {
    bool rotation = true;
    
    long n_vertices_per_small_particle = 26;
    long n_vertices_per_large_particle = 36;

    double count_ratio = 0.5;

    long n_small_particles = static_cast<long>(n_particles * count_ratio);
    long n_large_particles = n_particles - n_small_particles;
    long n_vertices = n_small_particles * n_vertices_per_small_particle + n_large_particles * n_vertices_per_large_particle;
    ConfigDict config;
    config["particle_type"] = "rigid";
    config["n_particles"] = n_particles;
    config["n_vertices"] = n_vertices;
    config["packing_fraction"] = packing_fraction;
    config["particle_dim_block"] = 256;
    config["vertex_dim_block"] = 256;
    config["segment_length_per_vertex_diameter"] = 1.0;
    config["vertex_radius"] = 0;
    config["n_vertices_per_small_particle"] = n_vertices_per_small_particle;
    config["n_vertices_per_large_particle"] = n_vertices_per_large_particle;
    config["size_ratio"] = 1.4;
    config["count_ratio"] = count_ratio;
    config["rotation"] = rotation;
    config["e_c"] = 1.0;
    config["n_c"] = 2.0;
    config["mass"] = 1.0;
    config["seed"] = -1;
    config["neighbor_list_config"] = {
        {"neighbor_cutoff_multiplier", 1.5},
        {"neighbor_displacement_multiplier", 0.2},
        {"num_particles_per_cell", 8.0},
        {"cell_displacement_multiplier", 0.5},
        {"vertex_neighbor_cutoff_multiplier", 1.5},
        {"vertex_neighbor_displacement_multiplier", 0.2},
        {"neighbor_list_update_method", "cell"}
    };
    return config;
}