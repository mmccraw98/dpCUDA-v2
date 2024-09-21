#ifndef LOG_GROUPS_H
#define LOG_GROUPS_H

#include <string>
#include <vector>
#include "log_manager.h"

// log-objects (log name list and step manager)

// make virtual
// add virtual log() function
// add other log groups:
// energy
// console
// state (trajectory and things like that)








class LogGroup {
public:
    LogGroup(std::vector<std::string> log_names, LogManagerConfig config);
    ~LogGroup();
    void check_log_status(long step);
    std::vector<std::string> get_log_names();
    bool should_log = false;

    std::vector<std::string> log_names;
    LogManager log_manager;

    virtual void log(long step) = 0;
};

// TODO: Log from_num_saves_log(long max_steps, long min_save_decade = 10, std::string save_style = "log");  TODO

LogGroup from_num_saves_lin(long num_saves, long max_steps);

#endif // LOG_GROUPS_H