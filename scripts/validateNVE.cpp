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
    std::string root_path = "/home/mmccraw/dev/data/25-02-01/validating-nve/rb-verlet-no-rotation/";
    std::string particle_type = "RigidBumpyNoRotation";

    long num_steps_base = 1e5;
    long num_saves = 1e4;
    double temperature = 1e-4;
    bool overwrite = true;

    double dt_start = 1e-2;
    double dt_end = 1e-5;
    long num_dt_steps = 3;

    long num_repeats = 5;

    for (long i = 0; i < num_dt_steps; i++) {
        double dt_dimless = std::pow(10, std::log10(dt_start) + i * (std::log10(dt_end) - std::log10(dt_start)) / (num_dt_steps));

        for (long j = 0; j < num_repeats; j++) {
            std::string output_path = root_path + std::to_string(dt_dimless) + "/" + std::to_string(j) + "/";
            long num_steps = num_steps_base * static_cast<long>(dt_start / dt_dimless);
            long save_every_N_steps = static_cast<long>(num_steps / num_saves);

            std::unique_ptr<Particle> particle = createParticle(32, 0.6, particle_type, true);
            particle->setRandomVelocities(temperature);

            std::cout << "TIME UNIT: " << particle->getTimeUnit() << std::endl;

            ConfigDict nve_config_dict = get_nve_config_dict(dt_dimless / particle->getTimeUnit());
            NVE nve(*particle, nve_config_dict);

            std::vector<std::string> init_names = particle->getFundamentalValues();
            std::vector<ConfigDict> log_group_configs = {
                config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T", "phi"}, 1e4, "console"),
                config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T", "phi"}, save_every_N_steps, "energy"),
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
        }
    }
    return 0;
}