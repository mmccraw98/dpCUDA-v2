#pragma once

#include "base/config.h"
#include "disk/config.h"
#include "rigid_bumpy/config.h"

inline DiskParticleConfigDict get_standard_disk_config(long n_particles, double packing_fraction) {
    BidispersityConfigDict bidispersity_config;
    bidispersity_config["size_ratio"] = 1.4;
    bidispersity_config["count_ratio"] = 0.5;

    PointNeighborListConfigDict standard_neighbor_list_config;
    standard_neighbor_list_config["neighbor_cutoff_multiplier"] = 1.5;
    standard_neighbor_list_config["neighbor_displacement_multiplier"] = 0.2;
    standard_neighbor_list_config["num_particles_per_cell"] = 8.0;
    standard_neighbor_list_config["cell_displacement_multiplier"] = 0.5;
    standard_neighbor_list_config["neighbor_list_update_method"] = "cell";

    DiskParticleConfigDict standard_disk_config;
    standard_disk_config["dispersity_config"] = bidispersity_config.to_nlohmann_json();
    standard_disk_config["neighbor_list_config"] = standard_neighbor_list_config.to_nlohmann_json();
    standard_disk_config["packing_fraction"] = packing_fraction;
    standard_disk_config["n_particles"] = n_particles;
    standard_disk_config["e_c"] = 1.0;
    standard_disk_config["n_c"] = 2.0;
    standard_disk_config["particle_dim_block"] = 256;
    
    // standard_disk_config["seed"] = -1;
    standard_disk_config["seed"] = 0;
    
    standard_disk_config["mass"] = 1.0;
    return standard_disk_config;
}

inline RigidBumpyParticleConfigDict get_standard_rigid_bumpy_config(long n_particles, double packing_fraction) {
    BidispersityConfigDict bidispersity_config;
    bidispersity_config["size_ratio"] = 1.4;
    bidispersity_config["count_ratio"] = 0.5;

    PointNeighborListConfigDict standard_neighbor_list_config;
    standard_neighbor_list_config["neighbor_cutoff_multiplier"] = 1.5;
    // standard_neighbor_list_config["neighbor_cutoff_multiplier"] = 4.0;

    standard_neighbor_list_config["neighbor_displacement_multiplier"] = 0.2;
    standard_neighbor_list_config["num_particles_per_cell"] = 8.0;

    standard_neighbor_list_config["cell_displacement_multiplier"] = 0.5;
    // standard_neighbor_list_config["cell_displacement_multiplier"] = 0.2;

    standard_neighbor_list_config["neighbor_list_update_method"] = "cell";

    standard_neighbor_list_config["vertex_neighbor_cutoff_multiplier"] = 1.5;
    // standard_neighbor_list_config["vertex_neighbor_cutoff_multiplier"] = 4.0;
    
    standard_neighbor_list_config["vertex_neighbor_displacement_multiplier"] = 0.2;

    RigidBumpyParticleConfigDict standard_rigid_bumpy_config;
    standard_rigid_bumpy_config["rotation"] = true;
    standard_rigid_bumpy_config["n_vertices_per_large_particle"] = 36;
    standard_rigid_bumpy_config["dispersity_config"] = bidispersity_config.to_nlohmann_json();
    standard_rigid_bumpy_config["neighbor_list_config"] = standard_neighbor_list_config.to_nlohmann_json();
    standard_rigid_bumpy_config["packing_fraction"] = packing_fraction;
    standard_rigid_bumpy_config["n_particles"] = n_particles;
    standard_rigid_bumpy_config["n_vertices"] = 0;
    standard_rigid_bumpy_config["e_c"] = 1.0;
    standard_rigid_bumpy_config["n_c"] = 2.0;
    standard_rigid_bumpy_config["particle_dim_block"] = 256;
    standard_rigid_bumpy_config["vertex_dim_block"] = 256;
    standard_rigid_bumpy_config["segment_length_per_vertex_diameter"] = 1.0;
    standard_rigid_bumpy_config["vertex_radius"] = 0;
    standard_rigid_bumpy_config["n_vertices_per_small_particle"] = 26;

    // standard_rigid_bumpy_config["seed"] = -1;
    standard_rigid_bumpy_config["seed"] = 0;
    
    standard_rigid_bumpy_config["mass"] = 1.0;
    return standard_rigid_bumpy_config;
}