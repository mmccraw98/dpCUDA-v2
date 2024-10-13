#ifndef ORCHESTRATOR_H
#define ORCHESTRATOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <unordered_set>
#include "../particle/particle.h"
#include "../integrator/integrator.h"
#include "utils.h"

class Orchestrator {
private:
    Particle& particle;  // the particle object (and integrator if needed) are private since the orchestrator handles everything
    Integrator* integrator;
    bool has_integrator;
    std::unordered_map<std::string, bool> pre_req_calculation_status;

public:
    Orchestrator(Particle& particle, Integrator* integrator = nullptr);  // can pass the integrator here too - this way, the log groups dont need to know about where the values are coming from
    ~Orchestrator();

    void init_pre_req_calculation_status();
    void handle_pre_req_calculations(const std::string& log_name);
    double apply_modifier(std::string& modifier, double value);
    std::vector<long> get_vector_size(const std::string& unmodified_log_name);
    
    template <typename T>
    T get_value(const std::string& unmodified_log_name, long step) {
        if (pre_req_calculation_status.find(unmodified_log_name) != pre_req_calculation_status.end()) {
            handle_pre_req_calculations(unmodified_log_name);
        }

        if (unmodified_log_name == "step") {
            return step;
        } else if (unmodified_log_name == "KE") {
            return particle.totalKineticEnergy();
        } else if (unmodified_log_name == "PE") {
            return particle.totalPotentialEnergy();
        } else if (unmodified_log_name == "TE") {
            return particle.totalEnergy();
        } else if (unmodified_log_name == "T") {
            return particle.calculateTemperature();
        } else {
            std::cerr << "Orchestrator::get_value: Log name not recognized: " << unmodified_log_name << std::endl;
            return 0.0;
        }
    }
    
    template <typename T>
    thrust::host_vector<T> get_vector_value(const std::string& unmodified_log_name) {
        return particle.getArray<T>(unmodified_log_name);
    }
};


//     def init_pre_req_calculation_status(self):
//         self.pre_req_calculation_status = {log_name: False for log_name in particle.pre_req_calculations}

//     def handle_pre_req_calculations(self, log_name):
//         if log_name == 'ke':  # put in the 'ingredients' needed for the determination of the variable
//             if not self.pre_req_calculation_status['ke']:
//                 self.particle.calc_kinetic_energy()  # there can be multiple ingredients
//             self.pre_req_calculation_status['ke'] = True
//         elif log_name == 'pe':
//             if not self.pre_req_calculation_status['pe']:
//                 self.particle.calc_potential_energy()
//             self.pre_req_calculation_status['pe'] = True
//         elif log_name == 't':
//             if not self.pre_req_calculation_status['ke']:
//                 self.particle.calc_kinetic_energy()
//             self.pre_req_calculation_status['ke'] = True
        
//     def get_value(self, log_name, step):  # define the value that is returned
//         if log_name in self.pre_req_calculation_status:
//             self.handle_pre_req_calculations(log_name)
//         if log_name == 'step':
//             return step
//         elif log_name == 'ke':
//             return self.particle.get_kinetic_energy()
//         elif log_name == 'pe':
//             return self.particle.get_potential_energy()
//         elif log_name == 't':
//             return self.particle.get_temperature()
//         else:
//             print(f"log name not recognized: {log_name}")
//             return None





// class Orchestrator {
// private:
//     std::vector<std::shared_ptr<LogGroup>> log_groups; // Stores all LogGroups

// public:
//     Orchestrator(const std::vector<std::shared_ptr<LogGroup>>& log_groups) : log_groups(log_groups) {}

//     std::unordered_set<std::string> get_unique_unmodified_log_names(long step) {
//         std::unordered_set<std::string> unique_log_names;

//         // Iterate through each log group and check if it's active for this step
//         for (const auto& log_group : log_groups) {
//             if (log_group->should_log(step)) {
//                 // Get the log names and insert the unmodified names into the set
//                 auto group_log_names = log_group->get_log_names();
//                 for (const auto& log_name : group_log_names) {
//                     unique_log_names.insert(get_unmodified_log_name(log_name));
//                 }
//             }
//         }

//         return unique_log_names; // This will contain only unique unmodified log names
//     }

    

// private:
//     std::string get_unmodified_log_name(const std::string& name) {
//         // Remove any modifiers like "/N", "/Nv", etc.
//         size_t pos = name.find('/');
//         return (pos != std::string::npos) ? name.substr(0, pos) : name;
//     }
// };


//     /**
//      * @brief Precalculates values needed for logging.
//      */
//     void precalculate();

//     /**
//      * @brief Gets the value of a log variable.
//      * @param unmodified_log_name The unmodified name of the log variable.
//      * @return The value of the log variable.
//      */
//     double get_value(const std::string& unmodified_log_name);

//     /**
//      * @brief Applies a modifier to a given value
//      * @param name The name of the variable to apply the modifier to
//      * @param value The value to apply the modifier to
//      * @return The modified value
//      */
//     double apply_modifier(std::string& name, double value);

#endif /* ORCHESTRATOR_H */