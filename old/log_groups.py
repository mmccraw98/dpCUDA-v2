import numpy as np
class Particle:
    pre_req_calculations = ['pe', 'ke']
    pos = np.array([1.0, 1.0, 1.0])
    vel = np.array([0.5, 0.5, 0.5])
    ke = np.array([0.0, 0.0, 0.0])
    N = 3
    mass = 1.0
    def calc_kinetic_energy(self):
        self.ke = 0.5 * self.mass * self.vel ** 2
    def calc_potential_energy(self):
        pass
    def get_kinetic_energy(self):
        return np.sum(self.ke)
    def get_temperature(self):
        return self.get_kinetic_energy() / (1.5 * self.mass)
    def get_potential_energy(self):
        return 0.0
    def get_position(self):
        return self.pos
    def get_velocity(self):
        return self.vel


# IMPLEMENTED
class Orchestrator:
    def __init__(self, particle):  # can pass the integrator here too - this way, the log groups dont need to know about where the values are coming from
        self.particle = particle
        self.pre_req_calculation_status = {}

    def init_pre_req_calculation_status(self):
        self.pre_req_calculation_status = {log_name: False for log_name in particle.pre_req_calculations}

    def handle_pre_req_calculations(self, log_name):
        if log_name == 'ke':  # put in the 'ingredients' needed for the determination of the variable
            if not self.pre_req_calculation_status['ke']:
                self.particle.calc_kinetic_energy()  # there can be multiple ingredients
            self.pre_req_calculation_status['ke'] = True
        elif log_name == 'pe':
            if not self.pre_req_calculation_status['pe']:
                self.particle.calc_potential_energy()
            self.pre_req_calculation_status['pe'] = True
        elif log_name == 't':
            if not self.pre_req_calculation_status['ke']:
                self.particle.calc_kinetic_energy()
            self.pre_req_calculation_status['ke'] = True
        
    def get_value(self, log_name, step):  # define the value that is returned
        if log_name in self.pre_req_calculation_status:
            self.handle_pre_req_calculations(log_name)
        if log_name == 'step':
            return step
        elif log_name == 'ke':
            return self.particle.get_kinetic_energy()
        elif log_name == 'pe':
            return self.particle.get_potential_energy()
        elif log_name == 't':
            return self.particle.get_temperature()
        else:
            print(f"log name not recognized: {log_name}")
            return None
        
# IMPLEMENTED
class LogManager:
    def __init__(self, save_style, save_freq, min_save_decade, reset_save_decade):
        self.save_style = save_style
        self.save_freq = save_freq
        self.min_save_decade = min_save_decade
        self.reset_save_decade = reset_save_decade
        self.multiple = 0
        self.decade = 10
        self.should_log = False
    
    def check_log_status(self, step):
        if self.save_style == "lin":
            self.should_log = step % self.save_freq == 0
        elif self.save_style == "log":
            if (step > self.multiple * self.reset_save_decade):
                self.save_freq = self.min_save_decade
                self.multiple += 1
            if (step - (self.multiple - 1) * self.reset_save_decade) > self.save_freq * self.decade:
                self.save_freq *= self.decade
            if (step - (self.multiple - 1) * self.reset_save_decade) % self.save_freq == 0:
                self.should_log = True
            else:
                self.should_log = False

# IMPLEMENTED
class LogGroup:
    def __init__(self, log_names, log_manager, orchestrator):
        self.log_names = log_names
        self.log_manager = log_manager
        self.orchestrator = orchestrator
    def update_log_status(self, step):
        self.log_manager.check_log_status(step)
    def log(self, step):
        print("NOT IMPLEMENTED")


# IMPLEMENTED
class MacroLog(LogGroup):
    modifier = "/"
    def __init__(self, log_names, log_manager, orchestrator):
        super().__init__(log_names, log_manager, orchestrator)
        self.unmodified_log_names = self.get_unmodified_log_names()
        self.delimeter = None
        self.precision = None
        self.width = None

    def get_unmodified_log_names(self):
        return [log_name.split(self.modifier)[0] for log_name in self.log_names]
    
    def log_name_is_modified(self, log_name):
        return self.modifier in log_name
    
    def apply_modifier(self, log_name, value):
        modifier = log_name.split(self.modifier)[1]
        if modifier == "N":
            return value / self.orchestrator.particle.N
        else:
            print(f"modifier not recognized: {modifier}")

    def log(self, step):
        print("NOT IMPLEMENTED")

# IMPLEMENTED
class EnergyLog(MacroLog):
    def __init__(self, log_names, log_manager, orchestrator):
        super().__init__(log_names, log_manager, orchestrator)
        self.has_header = False
        self.energy_file = ""
        self.delimeter = ","
        self.width = 10
        self.precision = 3

    def write_header(self):
        for i, log_name in enumerate(self.log_names):
            self.energy_file += log_name
            if i < len(self.log_names) - 1:
                self.energy_file += self.delimeter
        self.energy_file += "\n"

    def log(self, step):
        if not self.has_header:
            self.write_header()
            self.has_header = True
        for i, (log_name, unmodified_log_name) in enumerate(zip(self.log_names, self.unmodified_log_names)):
            value = self.orchestrator.get_value(unmodified_log_name, step)
            if self.log_name_is_modified(log_name):
                value = self.apply_modifier(log_name, value)
            value = float(value)
            self.energy_file += f"{value:{self.width}.{self.precision}}"
            if i < len(self.log_names) - 1:
                self.energy_file += self.delimeter
        self.energy_file += "\n"
    
# IMPLEMENTED
class ConsoleLog(MacroLog):
    def __init__(self, log_names, log_manager, orchestrator):
        super().__init__(log_names, log_manager, orchestrator)
        self.precision = 3
        self.width = 10
        self.last_header_log = 0
        self.header_log_freq = 10
        self.delimeter = "|"

    def write_header(self):
        out = ""
        out += "_" * (self.width * len(self.log_names) + (len(self.log_names) - 1))
        out += "\n"
        for i, log_name in enumerate(self.log_names):
            out += f"{log_name:{self.width}}"
            if i < len(self.log_names) - 1:
                out += self.delimeter
        out += "\n" + "_" * (self.width * len(self.log_names) + (len(self.log_names) - 1))
        out += "\n"
        print(out)

    def log(self, step):
        if self.last_header_log > self.header_log_freq:
            self.write_header()
            self.last_header_log = 0
        self.last_header_log += 1
        out = ""
        for i, (log_name, unmodified_log_name) in enumerate(zip(self.log_names, self.unmodified_log_names)):
            value = self.orchestrator.get_value(unmodified_log_name, step)
            if self.log_name_is_modified(log_name):
                value = self.apply_modifier(log_name, value)
            value = float(value)
            out += f"{value:{self.width}.{self.precision}}"
            if i < len(self.log_names) - 1:
                out += self.delimeter
        print(out)

particle = Particle()

orchestrator = Orchestrator(particle)

N = 10000
save_freq = 1  # save_freq does nothing in the log case
min_save_decade = 1  # the minimum decade of the save frequency
reset_save_decade = 1000  # the save frequency is reset at this decade
energy_log_manager = LogManager("log", save_freq=save_freq, min_save_decade=min_save_decade, reset_save_decade=reset_save_decade)
energy_log = EnergyLog(["step", "pe", "t"], energy_log_manager, orchestrator)

console_log_manager = LogManager("lin", 100, None, None)
console_log = ConsoleLog(["step", "ke", "pe", "t"], console_log_manager, orchestrator)

log_groups = [energy_log, console_log]

for step in range(1000):
    log_required = False
    for log_group in log_groups:
        log_group.update_log_status(step)
        if log_group.log_manager.should_log:
            log_required = True
    if log_required:
        orchestrator.init_pre_req_calculation_status()
        for log_group in log_groups:
            log_group.log(step)
print(len(energy_log.energy_file.split("\n")))