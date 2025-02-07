#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"

#include "../include/integrator/nve.h"

#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"

#include "../include/particles/factory.h"

#include "../include/particles/standard_configs.h"

#include "../include/integrator/adam.h"

#include "../include/routines/compression.h"

#include "../include/routines/initialization.h"

#include "../include/particles/factory.h"

int main() {
    std::unique_ptr<Particle> particle = createParticle(1024, 0.6, "RigidBumpy", true);

    long num_steps = 1e6;
    long save_every_N_steps = 1e3;
    double dt_dimless = 1e-2;
    double temperature = 1e-4;
    bool overwrite = true;
    std::string output_path = "/home/mmccraw/dev/data/25-02-01/comparing-energy-scales/rb/";
    
    particle->setRandomVelocities(temperature);

    std::cout << "TIME UNIT: " << particle->getTimeUnit() << std::endl;

    ConfigDict nve_config_dict = get_nve_config_dict(dt_dimless / particle->getTimeUnit());
    NVE nve(*particle, nve_config_dict);

    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<std::string> state_names = {"positions", "velocities", "force_pairs", "distance_pairs", "overlap_pairs", "radsum_pairs", "pair_separation_angle", "pair_ids"};
    if (particle->config.at("particle_type").get<std::string>() == "RigidBumpy") {
        state_names.push_back("angle_pairs_i");
        state_names.push_back("angle_pairs_j");
        state_names.push_back("this_vertex_contact_counts");
        state_names.push_back("vertex_positions");
        state_names.push_back("angles");
        state_names.push_back("angular_velocities");
    }
    std::vector<ConfigDict> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T", "phi"}, 1e4, "console"),
        config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T", "phi"}, save_every_N_steps, "energy"),
        // config_from_names_lin_everyN(state_names, save_every_N_steps, "state"),
        config_from_names_log(state_names, num_steps, 1e1, 1e1, "state"),
        config_from_names_lin_everyN(init_names, 1e4, "restart")
    };

    IOManager dynamics_io_manager(log_group_configs, *particle, &nve, output_path, 20, overwrite);
    dynamics_io_manager.write_params();

    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    long step = 0;
    while (step < num_steps) {
        nve.step();
        dynamics_io_manager.log(step);
        step++;
    }
    dynamics_io_manager.log(step, true);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}