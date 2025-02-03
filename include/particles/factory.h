#pragma once

#include "particles/base/particle.h"
#include "particles/disk/disk.h"
#include "particles/rigid_bumpy/rigid_bumpy.h"

#include "particles/base/config.h"
#include "particles/disk/config.h"
#include "particles/rigid_bumpy/config.h"

#include "io/io_utils.h"

inline std::unique_ptr<Particle> createParticle(ConfigDict& config) {
    std::string type_name = config["type_name"];
    std::unique_ptr<Particle> particle;
    if (type_name == "Disk") {
        particle = std::make_unique<Disk>();
    } else if (type_name == "RigidBumpy") {
        particle = std::make_unique<RigidBumpy>();
    }
    particle->initializeFromConfig(config);
    return particle;
}

inline std::tuple<std::shared_ptr<Particle>, long, std::filesystem::path, std::filesystem::path>
loadParticleFromRoot(const std::string& root, long trajectory_frame) 
{
    // Convert the root to a path
    std::filesystem::path root_path(root);

    // 1) Check for the system path
    std::filesystem::path system_path = get_path(root_path, "system");

    // 2) Check for the trajectory path
    std::filesystem::path trajectory_path = get_path(root_path, "trajectories");

    // 3) Get the particle config
    std::filesystem::path particle_config_path = get_path(system_path, "particle_config.json");
    ConfigDict particle_config;
    particle_config.from_json(particle_config_path);

    // 4) Create a unique_ptr<Particle> from the config
    auto particle_unique = createParticle(particle_config);

    // 5) Convert it to a shared_ptr<Particle> for our return type
    std::shared_ptr<Particle> particle(std::move(particle_unique));

    particle->initNeighborList();

    // 6) Open "init" and load everything in it, overwriting whatever is in the particle
    particle->loadDataFromPath(system_path / "init", ".dat");
    
    // 7) Open a given trajectory frame, load everything in it, overwrite whatever is in the particle
    auto frame_path_and_frame = get_trajectory_frame_path(trajectory_path, "t", trajectory_frame);
    std::filesystem::path frame_path = std::get<0>(frame_path_and_frame);
    long frame = std::get<1>(frame_path_and_frame);
    particle->loadDataFromPath(frame_path, ".dat");

    // 8) Call finalizeLoading for the particle
    particle->finalizeLoading();

    // 9) Set up neighbors again and calculate forces
    particle->updateNeighborList();
    particle->calculateForces();  // make sure forces are calculated before the integration
    double force_balance = particle->getForceBalance();
    if (force_balance / particle->n_particles / particle->e_c > 1e-14) {
        std::cout << "WARNING: Particle::setupNeighbors: Force balance is "
                  << force_balance << ", there will be an error!\n";
    }

    // TODO: SET THE PACKING FRACTION IN THE CONFIG TO BE THE NEW VALUE
    particle->calculateParticleArea();
    particle->config["packing_fraction"] = particle->getPackingFraction();

    // Finally, return the tuple:
    // (particle shared_ptr, frame number, system path, trajectory path)
    return {particle, frame, system_path, trajectory_path};
}