#include "../../include/routines/initialization.h"

std::tuple<SwapData2D<double>, SwapData1D<double>, SwapData1D<double>> get_minimal_overlap_positions_and_radii(const BaseParticleConfig& config) {
    std::cout << "Running Routine: get_minimal_overlap_positions_and_radii" << std::endl;
    auto particle = create_particle(config);

    particle->initAdamVariables();

    double alpha = 1e-4;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;
    long num_steps = 1e5;

    AdamConfig adam_config(alpha, beta1, beta2, epsilon);
    Adam adam(*particle, adam_config);

    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin_everyN({"step", "KE/N", "PE/N", "TE/N", "T"}, 1e3, "console"),  // logs to the console
    };
    IOManager io_manager(log_group_configs, *particle, &adam, "", 1, true);

    long step = 0;
    double dof = static_cast<double>(particle->n_dof);
    double last_avg_pe = 0.0;
    double avg_pe_diff = 0.0;
    while (step < num_steps) {
        adam.minimize(step);
        io_manager.log(step);
        double avg_pe = particle->totalPotentialEnergy() / dof;
        avg_pe_diff = std::abs(avg_pe - last_avg_pe);
        last_avg_pe = avg_pe;
        if (avg_pe_diff < avg_pe_diff_target || avg_pe < avg_pe_target) {
            break;
        }
        step++;
    }

    // copy the disk positions and radii and then use that to determine the rigid particle positions
    SwapData2D<double> positions;
    positions.copyFrom(particle->positions);
    SwapData1D<double> radii;
    radii.copyFrom(particle->radii);
    SwapData1D<double> box_size;
    box_size.copyFrom(particle->box_size);
    std::cout << "Finished Routine: get_minimal_overlap_positions_and_radii" << std::endl;

    return std::make_tuple(positions, radii, box_size);
}
