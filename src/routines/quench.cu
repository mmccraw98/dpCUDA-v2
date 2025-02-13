#include "quench.h"

void runNVTRescalingQuench(Particle& particle, double dt_dimless, long rescale_freq, double target_temperature, double temperature, double cooling_rate) {
    // if the target temperature and temperature are the same, do nothing
    if (target_temperature == temperature) {
        // however, check the real temperature, if it is sufficiently different to the target temperature, rescale to set the temperature
        double real_temperature = particle.calculateTemperature();
        if (std::abs(std::log10(real_temperature / target_temperature)) > 0.01) {
            particle.scaleVelocitiesToTemperature(target_temperature);
        }
    } else {
        // linearly ramp between the two
        ConfigDict nve_config_dict = get_nve_config_dict(dt_dimless / particle.getTimeUnit());
        NVE nve(particle, nve_config_dict);
        ConfigDict quench_console_log = config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T"}, 1e3, "console");
        std::vector<ConfigDict> log_group_configs = {quench_console_log};
        IOManager quench_io_manager(log_group_configs, particle, &nve, "", 20, true);
        double dt = nve_config_dict["dt"].get<double>();
        long num_ramp_steps = static_cast<long>(std::abs(temperature - target_temperature) / (cooling_rate * dt));
        if (num_ramp_steps == 0) {
            std::cout << "Warning: Cooling rate is too high, num_ramp_steps is 0, setting to 1" << std::endl;
            num_ramp_steps = 1;
        }
        double true_cooling_rate = (target_temperature - temperature) / (num_ramp_steps * dt);
        for (long i = 0; i < num_ramp_steps * 2; i++) {
            nve.step();
            if (i <= num_ramp_steps) {
                temperature += true_cooling_rate * dt;
            }
            if (i % rescale_freq == 0) {
                particle.scaleVelocitiesToTemperature(temperature);
            }
            quench_io_manager.log(i);
        }
    }
}