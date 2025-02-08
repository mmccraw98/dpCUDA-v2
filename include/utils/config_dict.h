#pragma once

#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <filesystem>

class ConfigDict : public nlohmann::json {
public:
    // Save to file
    void save(const std::string& path) const {
        std::ofstream ofs(path);
        if (!ofs) throw std::runtime_error("Failed to open file for writing: " + path);
        ofs << this->dump(4);
    }

    void save(const std::filesystem::path& path) const {
        save(path.string());
    }

    // Load from file
    void load(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) throw std::runtime_error("Failed to open file for reading: " + path);
        ifs >> *this;
    }

    void load(const std::filesystem::path& path) {
        load(path.string());
    }
};

inline ConfigDict load_config_dict(const std::filesystem::path& path) {
    ConfigDict config;
    config.load(path);
    return config;
}