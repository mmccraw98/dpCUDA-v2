#ifndef PARTICLE_CONFIG_H
#define PARTICLE_CONFIG_H

#include <iostream>
#include <nlohmann/json.hpp>

/**
 * @brief Base class for particle configuration.
 */
struct BaseParticleConfig {
    long seed;
    long n_particles;
    double mass;
    double e_c;
    double n_c;
    double packing_fraction;
    double neighbor_cutoff_multiplier;
    double neighbor_displacement_multiplier;
    double num_particles_per_cell;
    double cell_displacement_multiplier;
    std::string neighbor_list_update_method;
    long particle_dim_block;
    std::string type_name;
    std::string dispersity_type;
    
    /**
     * @brief Constructor for the base particle configuration.
     * 
     * @param seed The seed for the random number generator.
     * @param n_particles The number of particles.
     * @param mass The mass of the particles.
     * @param e_c The energy constant.
     * @param n_c The number constant.
     * @param packing_fraction The packing fraction of the particles.
     * @param neighbor_cutoff_multiplier Particles within this multiple of the maximum particle diameter will be considered neighbors.
     * @param neighbor_displacement_multiplier If the maximum displacement of a particle exceeds this multiple of the neighbor cutoff, the neighbor list will be updated.
     * @param num_particles_per_cell The desired number of particles per cell.
     * @param cell_displacement_multiplier The multiplier for the cell displacement in terms of the cell size.
     * @param neighbor_list_update_method The method for updating the neighbor list: "cell", "verlet", or "none"
     * @param particle_dim_block The number of threads in the block.
     */
    BaseParticleConfig(long seed, long n_particles, double mass, double e_c, double n_c, 
                    double packing_fraction, double neighbor_cutoff_multiplier, double neighbor_displacement_multiplier, double num_particles_per_cell, double cell_displacement_multiplier, std::string neighbor_list_update_method, long particle_dim_block)
        : seed(seed), n_particles(n_particles), mass(mass), e_c(e_c), n_c(n_c), 
        packing_fraction(packing_fraction), neighbor_cutoff_multiplier(neighbor_cutoff_multiplier), 
        neighbor_displacement_multiplier(neighbor_displacement_multiplier), num_particles_per_cell(num_particles_per_cell),
        cell_displacement_multiplier(cell_displacement_multiplier), neighbor_list_update_method(neighbor_list_update_method), particle_dim_block(particle_dim_block) {}

    virtual ~BaseParticleConfig() = default;

    /**
     * @brief Serialize the particle configuration to a JSON object.
     * 
     * @return The JSON object.
     */
    virtual nlohmann::json to_json() const {
        return nlohmann::json{
            {"seed", seed},
            {"n_particles", n_particles},
            {"mass", mass},
            {"e_c", e_c},
            {"n_c", n_c},
            {"packing_fraction", packing_fraction},
            {"neighbor_cutoff_multiplier", neighbor_cutoff_multiplier},
            {"neighbor_displacement_multiplier", neighbor_displacement_multiplier},
            {"num_particles_per_cell", num_particles_per_cell},
            {"cell_displacement_multiplier", cell_displacement_multiplier},
            {"neighbor_list_update_method", neighbor_list_update_method},
            {"particle_dim_block", particle_dim_block},
            {"type_name", type_name},
            {"dispersity_type", dispersity_type},
        };
    }

    /**
     * @brief Deserialize the particle configuration from a JSON object.
     * 
     * @param j The JSON object.
     * @return The particle configuration.
     */
    static BaseParticleConfig from_json(const nlohmann::json& j) {
        return BaseParticleConfig{
            j.at("seed").get<long>(),
            j.at("n_particles").get<long>(),
            j.at("mass").get<double>(),
            j.at("e_c").get<double>(),
            j.at("n_c").get<double>(),
            j.at("packing_fraction").get<double>(),
            j.at("neighbor_cutoff_multiplier").get<double>(),
            j.at("neighbor_displacement_multiplier").get<double>(),
            j.at("num_particles_per_cell").get<double>(),
            j.at("cell_displacement_multiplier").get<double>(),
            j.at("neighbor_list_update_method").get<std::string>(),
            j.at("particle_dim_block").get<long>(),
        };
    }
};

/**
 * @brief Configuration for bidisperse particles.
 */
struct BidisperseParticleConfig : public BaseParticleConfig {
    double size_ratio;  // The ratio of the radii of the large to small particles
    double count_ratio; // The ratio of the number of large to small particles

    /**
     * @brief Constructor for the bidisperse particle configuration.
     * 
     * @param seed The seed for the random number generator.
     * @param n_particles The number of particles.
     * @param mass The mass of the particles.
     * @param e_c The energy constant.
     * @param n_c The number constant.
     * @param packing_fraction The packing fraction of the particles.
     * @param neighbor_cutoff_multiplier Particles within this multiple of the maximum particle diameter will be considered neighbors.
     * @param neighbor_displacement_multiplier If the maximum displacement of a particle exceeds this multiple of the neighbor cutoff, the neighbor list will be updated.
     * @param num_particles_per_cell The desired number of particles per cell.
     * @param cell_displacement_multiplier The multiplier for the cell displacement in terms of the cell size.
     * @param neighbor_list_update_method The method for updating the neighbor list: "cell", "verlet", or "none"
     * @param particle_dim_block The number of threads in the block.
     * @param size_ratio The ratio of the radii of the large to small particles.
     * @param count_ratio The ratio of the number of large to small particles.
     */
    BidisperseParticleConfig(long seed, long n_particles, double mass, double e_c, double n_c,
                            double packing_fraction, double neighbor_cutoff_multiplier, double neighbor_displacement_multiplier, double num_particles_per_cell, double cell_displacement_multiplier, std::string neighbor_list_update_method, long particle_dim_block,
                            double size_ratio, double count_ratio)
        : BaseParticleConfig(seed, n_particles, mass, e_c, n_c, packing_fraction, neighbor_cutoff_multiplier, neighbor_displacement_multiplier, num_particles_per_cell, cell_displacement_multiplier, neighbor_list_update_method, particle_dim_block),
        size_ratio(size_ratio), count_ratio(count_ratio) {
            dispersity_type = "Bidisperse";
        }

    /**
     * @brief Serialize the bidisperse particle configuration to a JSON object.
     * 
     * @return The JSON object.
     */
    nlohmann::json to_json() const override {
        nlohmann::json j = BaseParticleConfig::to_json();  // Call base class serialization
        j["size_ratio"] = size_ratio;
        j["count_ratio"] = count_ratio;
        return j;
    }

    /**
     * @brief Deserialize the bidisperse particle configuration from a JSON object.
     * 
     * @param j The JSON object.
     * @return The bidisperse particle configuration.
     */
    static BidisperseParticleConfig from_json(const nlohmann::json& j) {
        // Call base class deserialization
        BaseParticleConfig base_config = BaseParticleConfig::from_json(j);
        
        // Extract the additional subclass fields
        double size_ratio = j.at("size_ratio").get<double>();
        double count_ratio = j.at("count_ratio").get<double>();

        // Construct and return the subclass object
        return BidisperseParticleConfig(
            base_config.seed, base_config.n_particles, base_config.mass, base_config.e_c, 
            base_config.n_c, base_config.packing_fraction, base_config.neighbor_cutoff_multiplier, 
            base_config.neighbor_displacement_multiplier, base_config.num_particles_per_cell, base_config.cell_displacement_multiplier, base_config.neighbor_list_update_method, base_config.particle_dim_block, size_ratio, count_ratio
        );
    }
};

#endif // PARTICLE_CONFIG_H