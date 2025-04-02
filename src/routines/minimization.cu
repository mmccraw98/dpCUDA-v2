#include "../../include/routines/minimization.h"

void minimizeAdam(Particle& particle, double alpha, double beta1, double beta2, double epsilon, double avg_pe_target, double avg_pe_diff_target, long num_steps, long log_every_n) {
    particle.initAdamVariables();

    ConfigDict adam_config = get_adam_config_dict(alpha, beta1, beta2, epsilon);
    Adam adam(particle, adam_config);

    std::vector<ConfigDict> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "phi"}, log_every_n, "console"),  // logs to the console
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
    io_manager.log(step, true);
    if (step >= num_steps) {
        std::cout << "Warning: minimization did not converge" << std::endl;
    }
}

// Overloaded function for default parameters
void minimizeAdam(Particle& particle) {
    double alpha = 1e-4;  // probably best to be 1e-5
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;
    long num_steps = 1e5;
    long log_every_n = 1e3;
    minimizeAdam(particle, alpha, beta1, beta2, epsilon, avg_pe_target, avg_pe_diff_target, num_steps, log_every_n);
}

// Overloaded function for just passing adam parameters and num_steps
void minimizeAdam(Particle& particle, double alpha, double beta1, double beta2, double epsilon, long num_steps, long log_every_n) {
    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;
    minimizeAdam(particle, alpha, beta1, beta2, epsilon, avg_pe_target, avg_pe_diff_target, num_steps, log_every_n);
}

// Overloaded function for just passing log_every_n
void minimizeAdam(Particle& particle, long log_every_n) {
    double alpha = 1e-5;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    long num_steps = 1e5;
    minimizeAdam(particle, alpha, beta1, beta2, epsilon, num_steps, log_every_n);
}

void minimizeAdam(Particle& particle, double avg_pe_target, double avg_pe_diff_target) {
    double alpha = 1e-5;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    long num_steps = 1e5;
    long log_every_n = 1e3;
    minimizeAdam(particle, alpha, beta1, beta2, epsilon, avg_pe_target, avg_pe_diff_target, num_steps, log_every_n);
}

void minimizeFire(Particle& particle, double alpha_init, double dt, double avg_pe_target, double avg_pe_diff_target, long num_steps, long log_every_n) {
    ConfigDict fire_config = get_fire_config_dict(alpha_init, dt);
    Fire fire(particle, fire_config);

    std::vector<ConfigDict> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "KE/N", "phi"}, log_every_n, "console"),  // logs to the console
    };
    IOManager io_manager(log_group_configs, particle, &fire, "", 1, true);

    long step = 0;
    // double dof = static_cast<double>(particle.n_dof);
    double dof = static_cast<double>(particle.n_particles);
    double avg_pe_diff = 1e9;

    // start with the velocities set to zero and the forces calculated
    particle.setVelocitiesToZero();
    particle.zeroForceAndPotentialEnergy();  // not needed since forces are calculated in place
    particle.checkForNeighborUpdate();
    particle.calculateForces();
    double last_avg_pe = particle.totalPotentialEnergy() / dof;

    while (step < num_steps) {
        fire.step();
        io_manager.log(step);
        double avg_pe = particle.totalPotentialEnergy() / dof;
        avg_pe_diff = std::abs(avg_pe - last_avg_pe);
        last_avg_pe = avg_pe;
        if (avg_pe_diff < avg_pe_diff_target || avg_pe < avg_pe_target || fire.stopped) {
            if (avg_pe_diff < avg_pe_diff_target) {
                std::cout << "Broken due to pe diff" << std::endl;
            }
            break;
        }
        step++;
    }
    io_manager.log(step, true);
    if (step >= num_steps) {
        std::cout << "Warning: minimization did not converge" << std::endl;
    }
    particle.setVelocitiesToZero();
}

void minimizeFire(Particle& particle, double alpha_init, double dt, long num_steps, long log_every_n) {
    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;
    minimizeFire(particle, alpha_init, dt, avg_pe_target, avg_pe_diff_target, num_steps, log_every_n);
}

void minimizeFire(Particle& particle, long log_every_n) {
    double alpha_init = 0.1;
    double dt = 1e-2;
    long num_steps = 1e5;
    minimizeFire(particle, alpha_init, dt, num_steps, log_every_n);
}

void minimizeFire(Particle& particle, double avg_pe_target, double avg_pe_diff_target) {
    double alpha_init = 0.1;
    double dt = 1e-2;
    long num_steps = 1e5;
    long log_every_n = 1e3;
    minimizeFire(particle, alpha_init, dt, avg_pe_target, avg_pe_diff_target, num_steps, log_every_n);
}

void minimizeFire(Particle& particle) {
    double alpha_init = 0.1;
    double dt = 1e-2;
    long num_steps = 1e5;
    long log_every_n = 1e3;
    minimizeFire(particle, alpha_init, dt, num_steps, log_every_n);
}