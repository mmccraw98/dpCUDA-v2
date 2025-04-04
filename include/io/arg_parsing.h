#include <iostream>
#include <filesystem>
#include "../utils/config_dict.h"
#include "../particles/base/particle.h"


inline std::tuple<std::unique_ptr<Particle>, long, ConfigDict, ConfigDict, ConfigDict, ConfigDict> load_configs(int argc, char** argv, bool use_restart = true) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <script_path>" << std::endl;
        exit(1);
    }
    // convert the argv[1] to a path
    std::filesystem::path root = argv[1];
    std::filesystem::path run_config_path = root / "system" / "run_config.json";

    ConfigDict run_config = load_config_dict(run_config_path);
    bool minimize = run_config["minimize"];

    std::filesystem::path input_path = std::filesystem::path(run_config["input_dir"]);
    std::filesystem::path output_path = std::filesystem::path(run_config["output_dir"]);

    // load the logging configs
    ConfigDict console_config;
    ConfigDict energy_config;
    ConfigDict state_config;
    std::filesystem::path config_root;

    std::string load_source;
    if (use_restart) {
        load_source = "restart";
    } else {
        load_source = "init";
    }

    // load the particle and step
    std::unique_ptr<Particle> particle;
    long step = 0;
    if (input_path.empty() && !output_path.empty()) {
        ConfigDict particle_config = load_config_dict(output_path / "system" / "particle_config.json");
        // if the input path is not defined, create a new particle using the particle config in the output path
        particle = createParticle(particle_config, minimize);
        config_root = output_path;
    } else if (!input_path.empty() && !output_path.empty()) {
        // if the input path is defined and the output path is defined, load the particle from the input system path
        // load particle
        config_root = output_path;
        std::tie(particle, step) = loadParticle(input_path, load_source, -2);
        // do not resume the run
        step = 0;
    } else if (!input_path.empty() && output_path.empty()) {
        // if the input path is defined and the output path is not defined, resume the run from the last step
        // load particle and step
        config_root = input_path;
        std::tie(particle, step) = loadParticle(input_path, load_source, -2);
        // use the step from the input path
    } else {
        throw std::invalid_argument("Run config must define either an input or output path!  run_config: " + run_config.dump(4));
    }
    console_config.load(config_root / "system" / "console_log_config.json");
    energy_config.load(config_root / "system" / "energy_log_config.json");
    state_config.load(config_root / "system" / "state_log_config.json");
    std::cout << "Done loading data" << std::endl;
    return std::make_tuple(std::move(particle), step, console_config, energy_config, state_config, run_config);
}