#include "../../include/routines/initialization.h"

std::tuple<SwapData2D<double>, SwapData1D<double>, SwapData1D<double>> get_minimal_overlap_disk_positions_and_radii(ConfigDict& config, double overcompression_factor) {
    std::cout << "Running Routine: get_minimal_overlap_disk_positions_and_radii" << std::endl;
    config["packing_fraction"] += overcompression_factor;

    DiskParticleConfigDict disk_config;
    disk_config["seed"] = config["seed"];
    disk_config["n_particles"] = config["n_particles"];
    disk_config["mass"] = config["mass"];
    disk_config["e_c"] = config["e_c"];
    disk_config["n_c"] = config["n_c"];
    disk_config["packing_fraction"] = config["packing_fraction"];
    disk_config["neighbor_list_config"] = config["neighbor_list_config"];

    Disk disk;
    disk.initializeFromConfig(disk_config);

    disk.initAdamVariables();

    double alpha = 1e-4;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;
    long num_steps = 1e5;

    AdamConfigDict adam_config;
    adam_config["alpha"] = alpha;
    adam_config["beta1"] = beta1;
    adam_config["beta2"] = beta2;
    adam_config["epsilon"] = epsilon;
    Adam adam(disk, adam_config);

    std::vector<LogGroupConfigDict> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "phi"}, 1e3, "console"),  // logs to the console
    };
    IOManager io_manager(log_group_configs, disk, &adam, "", 1, true);

    long step = 0;
    double dof = static_cast<double>(disk.n_dof);
    double last_avg_pe = 0.0;
    double avg_pe_diff = 0.0;
    while (step < num_steps) {
        adam.minimize(step);
        io_manager.log(step);
        double avg_pe = disk.totalPotentialEnergy() / dof;
        avg_pe_diff = std::abs(avg_pe - last_avg_pe);
        last_avg_pe = avg_pe;
        if (avg_pe_diff < avg_pe_diff_target || avg_pe < avg_pe_target) {
            break;
        }
        step++;
    }

    // scale the disk positions and radii back down to the target packing fraction
    double packing_fraction = config["packing_fraction"];
    disk.scaleToPackingFraction(packing_fraction - overcompression_factor);

    // copy the disk positions and radii and then use that to determine the rigid particle positions
    SwapData2D<double> positions;
    positions.copyFrom(disk.positions);
    SwapData1D<double> radii;
    radii.copyFrom(disk.radii);
    SwapData1D<double> box_size;
    box_size.copyFrom(disk.box_size);
    std::cout << "Finished Routine: get_minimal_overlap_disk_positions_and_radii" << std::endl;

    return std::make_tuple(positions, radii, box_size);
}
