#include "../include/constants.h"
#include "../include/particle/particle.h"
#include "../include/particle/disk.h"
#include "../include/integrator/nve.h"
#include "../include/io/orchestrator.h"
#include "../include/particle/particle_factory.h"
#include "../include/io/utils.h"
#include "../include/io/console_log.h"
#include "../include/io/energy_log.h"
#include "../include/io/io_manager.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <nlohmann/json.hpp>

int main() {
    // Create particle configuration using BidisperseDiskConfig
    BidisperseDiskConfig config(0, 1024, 1.0, 1.0, 2.0, 0.5, 1.5, 256, 1.4, 0.5);

    // Save the configuration to a JSON file
    write_json_to_file("/home/mmccraw/dev/dpCUDA/old/system/config.json", config.to_json());

    // Create a particle instance using the particle factory
    auto particle = create_particle(config);  // Specify the particle type explicitly

    // Set random velocities for the particle
    particle->setRandomVelocities(1e-3);

    // Calculate time step for the NVE integrator
    double dt = 1e-2 * particle->getTimeUnit();

    // Instantiate the NVE integrator with the created particle
    NVE<Disk> nve(*particle, dt);

    // Set up log group configurations for the IOManager
    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin({"step", "TE", "KE", "PE", "T"}, 1e4, 100, "energy"),
        config_from_names_lin_everyN({"step", "T", "TE/N"}, 1e2, "console")
    };

    // Instantiate IOManager with the particle, NVE integrator, and log group configs
    IOManager<Disk, NVE<Disk>> io_manager(*particle, &nve, log_group_configs, "/home/mmccraw/dev/dpCUDA/old", false);

    // Write log group configurations to file
    io_manager.write_log_configs("/home/mmccraw/dev/dpCUDA/old");

    // Perform initial logging at step 0
    io_manager.log(0);


    // Example simulation loop (uncomment to use)
    long step = 0;
    while (step < 1e4) {
        // Integrator performs one simulation step
        nve.step();
        // Log data for the current step
        io_manager.log(step);
        step++;
    }

    return 0;
}
