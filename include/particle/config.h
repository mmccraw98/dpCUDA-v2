#ifndef PARTICLE_CONFIG_H
#define PARTICLE_CONFIG_H

#include <nlohmann/json.hpp>

struct BaseParticleConfig {
    long seed;
    long n_particles;
    double mass;
    double e_c;
    double n_c;
    double packing_fraction;
    double neighbor_cutoff;
    long dim_block;
    std::string type_name;
    std::string dispersity_type;
    
    // Constructor
    BaseParticleConfig(long seed, long n_particles, double mass, double e_c, double n_c, 
                    double packing_fraction, double neighbor_cutoff, long dim_block)
        : seed(seed), n_particles(n_particles), mass(mass), e_c(e_c), n_c(n_c), 
        packing_fraction(packing_fraction), neighbor_cutoff(neighbor_cutoff), dim_block(dim_block) {}

    virtual ~BaseParticleConfig() = default;

    // Serialize to JSON using nlohmann::json
    virtual nlohmann::json to_json() const {
        return nlohmann::json{
            {"seed", seed},
            {"n_particles", n_particles},
            {"mass", mass},
            {"e_c", e_c},
            {"n_c", n_c},
            {"packing_fraction", packing_fraction},
            {"neighbor_cutoff", neighbor_cutoff},
            {"dim_block", dim_block},
        };
    }

    // Deserialize from JSON using nlohmann::json
    static BaseParticleConfig from_json(const nlohmann::json& j) {
        return BaseParticleConfig{
            j.at("seed").get<long>(),
            j.at("n_particles").get<long>(),
            j.at("mass").get<double>(),
            j.at("e_c").get<double>(),
            j.at("n_c").get<double>(),
            j.at("packing_fraction").get<double>(),
            j.at("neighbor_cutoff").get<double>(),
            j.at("dim_block").get<long>(),
        };
    }
};

struct BidisperseParticleConfig : public BaseParticleConfig {
    double size_ratio;
    double count_ratio;
    std::string dispersity_type = "bidisperse";

    // Constructor for the subclass, calling the base class constructor
    BidisperseParticleConfig(long seed, long n_particles, double mass, double e_c, double n_c,
                            double packing_fraction, double neighbor_cutoff, long dim_block,
                            double size_ratio, double count_ratio)
        : BaseParticleConfig(seed, n_particles, mass, e_c, n_c, packing_fraction, neighbor_cutoff, dim_block),
        size_ratio(size_ratio), count_ratio(count_ratio) {}

    // Override to_json to serialize additional fields
    nlohmann::json to_json() const override {
        nlohmann::json j = BaseParticleConfig::to_json();  // Call base class serialization
        j["size_ratio"] = size_ratio;
        j["count_ratio"] = count_ratio;
        j["dispersity_type"] = dispersity_type;
        return j;
    }

    // Static method to deserialize and create a BidisperseParticleConfig instance
    static BidisperseParticleConfig from_json(const nlohmann::json& j) {
        // Call base class deserialization
        BaseParticleConfig base_config = BaseParticleConfig::from_json(j);
        
        // Extract the additional subclass fields
        double size_ratio = j.at("size_ratio").get<double>();
        double count_ratio = j.at("count_ratio").get<double>();

        // Construct and return the subclass object
        return BidisperseParticleConfig(
            base_config.seed, base_config.n_particles, base_config.mass, base_config.e_c, 
            base_config.n_c, base_config.packing_fraction, base_config.neighbor_cutoff, 
            base_config.dim_block, size_ratio, count_ratio
        );
    }
};

#endif // PARTICLE_CONFIG_H