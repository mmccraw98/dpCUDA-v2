#pragma once

#include <iostream>
#include <nlohmann/json.hpp>

#include <unordered_map>
#include <fstream>



struct ConfigDict {
    std::unordered_map<std::string, nlohmann::json> data;

    nlohmann::json to_nlohmann_json() const {
        return nlohmann::json(data);
    }

    // Load from JSON file
    bool from_json(const std::string& path) {
        try {
            std::ifstream file(path);
            if (!file.is_open()) {
                return false;
            }
            
            nlohmann::json j;
            file >> j;
            
            data.clear();
            for (const auto& [key, value] : j.items()) {
                data[key] = value;
            }
            
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }

    // Save to JSON file
    bool to_json(const std::string& path) const {
        try {
            nlohmann::json j(data);
            
            std::ofstream file(path);
            if (!file.is_open()) {
                return false;
            }
            
            file << j.dump(4);
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }

    // Convenience methods
    template<typename T>
    void insert(const std::string& key, const T& value) {
        if constexpr (std::is_base_of_v<ConfigDict, T>) {
            // Special handling for ConfigDict and its derived classes
            data[key] = value.to_nlohmann_json();
        } else {
            // Normal handling for other types
            data[key] = value;
        }
    }

    nlohmann::json& operator[](const std::string& key) {
        return data[key];
    }

    const nlohmann::json& operator[](const std::string& key) const {
        return data.at(key);
    }

    bool contains(const std::string& key) const {
        return data.count(key) > 0;
    }

    // Universal getter method for any type
    template<typename T>
    T get(const std::string& key, const T& default_value = T()) const {
        auto it = data.find(key);
        if (it != data.end()) {
            try {
                return it->second.get<T>();
            } catch (...) {
                return default_value;
            }
        }
        return default_value;
    }

    
};