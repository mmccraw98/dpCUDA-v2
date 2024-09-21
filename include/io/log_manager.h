#ifndef LOG_MANAGER_H
#define LOG_MANAGER_H

#include <string>
#include <vector>

// manages the step saving schemes in the integrator

struct LogManagerConfig {
    std::string save_style = "lin";
    long save_freq = 100;
    long min_save_decade = 10;
};

class LogManager {
private:
    long save_freq_multiple;
    long decade = 10;

public:
    LogManager(LogManagerConfig config);
    ~LogManager();

    LogManagerConfig config;
    
    bool should_log(long step);

};


#endif /* LOG_MANAGER_H */