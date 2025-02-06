#include "../../include/routines/minimization.h"

void minimizeAdam(Particle& particle) {
    particle.initAdamVariables();

    double alpha = 1e-4;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;
    long num_steps = 1e5;

    ConfigDict adam_config = get_adam_config_dict(alpha, beta1, beta2, epsilon);
    Adam adam(particle, adam_config);

    std::vector<ConfigDict> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "phi"}, 1e3, "console"),  // logs to the console
    };
    IOManager io_manager(log_group_configs, particle, &adam, "", 1, true);

    long step = 0;
    double dof = static_cast<double>(particle.n_dof);
    double last_avg_pe = 0.0;
    double avg_pe_diff = 0.0;
    while (step < num_steps) {
        adam.minimize(step);
        io_manager.log(step);
        double avg_pe = particle.totalPotentialEnergy() / dof;
        avg_pe_diff = std::abs(avg_pe - last_avg_pe);
        last_avg_pe = avg_pe;
        if (avg_pe_diff < avg_pe_diff_target || avg_pe < avg_pe_target) {
            break;
        }
        step++;
    }    
}