#pragma once

#include "../../utils/config_dict.h"

struct BidispersityConfigDict : public ConfigDict {
public:
    BidispersityConfigDict() {
        insert("type_name", "Bidisperse");
        insert("size_ratio", 0.0);
        insert("count_ratio", 0.0);
    }
};

struct BaseParticleConfigDict : public ConfigDict {
public:
    BaseParticleConfigDict() {
        insert("type_name", "Base");
        insert("seed", 0);
        insert("mass", 0.0);
        insert("packing_fraction", 0.0);
        insert("n_particles", 0);
        insert("e_c", 0.0);
        insert("n_c", 0.0);
        insert("particle_dim_block", 0);
        insert("dispersity_config", ConfigDict());
        insert("neighbor_list_config", ConfigDict());
    }
};

struct PointNeighborListConfigDict : public ConfigDict {
public:
    PointNeighborListConfigDict() {
        insert("type_name", "Point");
        insert("neighbor_cutoff_multiplier", 0.0);
        insert("neighbor_displacement_multiplier", 0.0);
        insert("num_particles_per_cell", 0.0);
        insert("cell_displacement_multiplier", 0.0);
        insert("neighbor_list_update_method", "none");
    }
};

struct PointParticleConfigDict : public BaseParticleConfigDict {
public:
    PointParticleConfigDict() {
        insert("type_name", "Point");
        insert("neighbor_list_config", ConfigDict());
    }
};

struct VertexNeighborListConfigDict : public PointNeighborListConfigDict {
public:
    VertexNeighborListConfigDict() {
        insert("type_name", "Vertex");
        insert("vertex_neighbor_cutoff_multiplier", 1.5);
        insert("vertex_neighbor_displacement_multiplier", 0.5);
    }
};

struct RigidVertexParticleConfigDict : public PointParticleConfigDict {
public:
    RigidVertexParticleConfigDict() {
        insert("type_name", "RigidVertex");
        insert("n_vertices_per_small_particle", 0);
        insert("n_vertices_per_large_particle", 0);
        insert("n_vertices", 0);
        insert("vertex_radius", 0.0);
        insert("segment_length_per_vertex_diameter", 0.0);
        insert("vertex_dim_block", 0);
        insert("rotation", false);
        insert("neighbor_list_config", ConfigDict());
    }
};