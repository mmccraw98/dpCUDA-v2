//

#ifndef FILEIO_H
#define FILEIO_H

#include "DPM2D.h"
#include "defs.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <experimental/filesystem>
#include <filesystem>

using namespace std;

/**
 * @brief Make directories if they don't exist
 * 
 * @param dirName Name of directory to make
 * @param warn If true and dirName already exists, the program will crash; otherwise, it will overwrite
 */
inline void makeDir(const std::string& dirName, bool warn = true) {
    namespace fs = std::filesystem; // Alias for easier use

    if (fs::exists(dirName)) {
        if (warn) {
            std::cerr << "Directory " << dirName << " already exists. Exiting." << std::endl;
            exit(1);
        }
        // If warn is false and the directory exists, do nothing
    } else {
        try {
            fs::create_directories(dirName); // Creates all necessary directories
            // std::cout << "Directory " << dirName << " created successfully." << std::endl;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error creating directory " << dirName << ": " << e.what() << std::endl;
            exit(1);
        }
    }
}

class ioDPMFile {
    public:
    ifstream inputFile;
    ofstream outputFile;
    ofstream energyFile;
    ofstream corrFile;
    DPM2D * dpm_;
    std::vector<std::string>& energyNames;
    std::vector<std::string>& stateNames;
    std::vector<std::string>& consoleNames;

    long logs_since_last_header = 0;

    // Default constructor declaration, if needed
    ioDPMFile() = delete; // Deleted if there is no meaningful default construction for this object

    // Corrected constructor using initializer list for references
    ioDPMFile(DPM2D* dpmPtr, std::vector<std::string>& energyNames, std::vector<std::string>& stateNames, std::vector<std::string>& consoleNames) 
    : dpm_(dpmPtr), energyNames(energyNames), stateNames(stateNames), consoleNames(consoleNames) {
        // No need to assign references in the constructor's body
    }

    void setEnergyNames(std::vector<std::string>& energyNames) {
        this->energyNames = energyNames;
    }

    void setStateNames(std::vector<std::string>& stateNames) {
        this->stateNames = stateNames;
    }
    
    void setConsoleNames(std::vector<std::string>& consoleNames) {
        this->consoleNames = consoleNames;
    }

    std::string getExperimentName(string prefix="") {
        std::string experiment_name = prefix;
        std::ostringstream particle_numbers;
        particle_numbers << "N_" << dpm_->getNumParticles() << "_nVpP_" << dpm_->getNumVertexPerParticle();
        experiment_name += dpm_->getParticleTypeString() + "_" + dpm_->getSmoothnessString() + "_" + dpm_->getDeformabilityString() + "_" + dpm_->getPotentialTypeString() + "_" + particle_numbers.str();
        return experiment_name;
    }

    void readPackingFromCellFormat(string fileName, long skiplines) {
        this->openInputFile(fileName);
        this->readFromCellFormat(skiplines);
    }

    // open file and check if it throws an error
    void openInputFile(string fileName) {
        inputFile = ifstream(fileName.c_str());
        if (!inputFile.is_open()) {
            cerr << "ioDPMFile::openInputFile: error: could not open input file " << fileName << endl;
            exit(1);
        }
    }

    void openOutputFile(string fileName) {
        outputFile = ofstream(fileName.c_str());
        if (!outputFile.is_open()) {
            cerr << "ioDPMFile::openOutputFile: error: could not open input file " << fileName << endl;
            exit(1);
        }
    }

    void openEnergyFile(const std::string& fileName, bool resume = false) {
        std::ifstream checkFile(fileName);
        std::string firstLine;
        bool addHeader = true;

        if (checkFile.is_open()) {
            if (std::getline(checkFile, firstLine)) {
                if (firstLine.find("step") != std::string::npos) {
                    addHeader = false;
                }
            }
            checkFile.close();
        }

        // Open the file with ofstream in append mode if resume is true
        if (resume) {
            energyFile.open(fileName.c_str(), std::ios_base::app);
        } else {
            energyFile.open(fileName.c_str());
        }

        if (!energyFile.is_open()) {
            std::cerr << "ioDPMFile::openEnergyFile: error: could not open energy file " << fileName << std::endl;
            exit(1);
        }

        // Ensure the file is opened in append mode when resume is true
        if (resume && energyFile.tellp() == std::streampos(0)) {
            std::cerr << "ioDPMFile::openEnergyFile: error: failed to open energy file in append mode " << fileName << std::endl;
            exit(1);
        }

        if (!resume && addHeader) {
            std::cout << "ioDPMFile::openEnergyFile: adding header to energy file" << std::endl;
            writeEnergyFileHeader();
        }
    }

    void writeConsoleHeader(long width = 12) {
        // get length of consoleNames
        long numNames = consoleNames.size();
        long totalWidth = (width + 3) * (numNames + 2);
        // write the header
        std::cout << std::string(totalWidth, '_') << std::endl;
        // std::cout << std::string((width + 8) * numNames + numNames, '_') << std::endl;
        std::cout << std::setw(width) << "step" << " | " << std::setw(width) << "time" << " | ";
        for (long i = 0; i < numNames; i++) {
            std::cout << std::setw(width) << consoleNames[i] << " | ";
        }
        std::cout << std::endl;
        // std::cout << std::string((width + 8) * numNames + numNames, '_') << std::endl;
        std::cout << std::string(totalWidth, '_') << std::endl;
    }

    void writeConsoleDataByName(const std::string& name, long width, long precision) {
        switch (dpm_->getDeformability()) {
            case simControlStruct::deformabilityEnum::rigid:
                if (name == "pe") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getParticleEnergy() << " | ";
                } else if (name == "ke") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getRigidKineticEnergy() << " | ";
                } else if (name == "te") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getRigidKineticEnergy() + dpm_->getParticleEnergy() << " | ";

                } else if (name == "temp_nr") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getRigidTemperature_noRotation() << " | ";

                } else if (name == "eta") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->eta << " | ";
                } else if (name == "pi") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->pi << " | ";
                } else if (name == "temp0") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->T0 << " | ";
                } else if (name == "nh_E_ext") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getNoseHooverExtendedEnergy(true) << " | ";
                } else if (name == "nh_E_ext_nr") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getNoseHooverExtendedEnergy(false) << " | ";

                } else if (name == "temp") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getRigidTemperature() << " | ";
                } else if (name == "phi") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getPhi() << " | ";
                } else if (name == "pressure") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getPressure() << " | ";
                } else if (name == "pe_v") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getVertexEnergy() << " | ";
                } else {
                    std::cerr << "FileIO::writeConsoleDataByName " << name << " not recognized" << std::endl;
                    exit(1);
                }
                break;

            case simControlStruct::deformabilityEnum::deformable:
                if (name == "pe") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getPotentialEnergy() << " | ";
                } else if (name == "ke") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getKineticEnergy() << " | ";
                } else if (name == "te") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getKineticEnergy() + dpm_->getPotentialEnergy() << " | ";
                } else if (name == "temp") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getTemperature() << " | ";
                } else if (name == "phi") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getPhi() << " | ";

                } else if (name == "eta") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->eta << " | ";
                } else if (name == "pi") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->pi << " | ";
                } else if (name == "temp0") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->T0 << " | ";
                } else if (name == "nh_E_ext") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getNoseHooverExtendedEnergy(false) << " | ";
                } else if (name == "nh_E_ext_nr") {
                    std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << dpm_->getNoseHooverExtendedEnergy(false) << " | ";

                } else {
                    std::cerr << "FileIO::writeConsoleDataByName " << name << " not recognized" << std::endl;
                    exit(1);
                }
                break;

            default:
                std::cerr << "error: particle type not recognized" << std::endl;
                exit(1);
        }
    }

    void writeConsoleData(long step, long width = 12, long precision = 3) {
        if ((logs_since_last_header > 10) || (step == 0)) {
            writeConsoleHeader(width);
            logs_since_last_header = 0;
        }
        logs_since_last_header += 1;
        std::cout << std::setw(width) << std::fixed << std::setprecision(precision) << step << " | " 
                << std::setw(width) << std::scientific << std::setprecision(precision) << (step * dpm_->dt) << " | ";
        for (long i = 0; i < consoleNames.size(); i++) {
            writeConsoleDataByName(consoleNames[i], width, precision);
        }
        std::cout << std::endl;
    }

    void openCorrFile(string fileName) {
        corrFile = ofstream(fileName.c_str());
        if (!corrFile.is_open()) {
        cerr << "ioDPMFile::openCorrFile: error: could not open input file " << fileName << endl;
        exit(1);
        }
    }

    void closeEnergyFile() {
        energyFile.close();
    }

    void closeCorrFile() {
        corrFile.close();
    }

    void saveCorr(long step, double timeStep) {
        double isf, visf, isfSq, visfSq, deltaA;
        isf = dpm_->getParticleISF(dpm_->getDeformableWaveNumber());
        visf = dpm_->getVertexISF();
        deltaA = dpm_->getAreaFluctuation();
        corrFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
        corrFile << setprecision(precision) << dpm_->getParticleMSD() << "\t";
        corrFile << setprecision(precision) << dpm_->getVertexMSD() << "\t";
        corrFile << setprecision(precision) << isf << "\t";
        corrFile << setprecision(precision) << visf << "\t";
        corrFile << setprecision(precision) << deltaA << endl;
    }

    thrust::host_vector<double> read1DIndexFile(string fileName, long numRows) {
        thrust::host_vector<long> data;
        this->openInputFile(fileName);
        string inputString;
        long tmp;
        for (long row = 0; row < numRows; row++) {
        getline(inputFile, inputString);
        sscanf(inputString.c_str(), "%ld", &tmp);
        data.push_back(tmp);
        }
        inputFile.close();
        return data;
    }

    void save1DIndexFile(string fileName, thrust::host_vector<long> data) {
        this->openOutputFile(fileName);
        long numRows = data.size();
        for (long row = 0; row < numRows; row++) {
        //sprintf(outputFile, "%ld \n", data[row]);
        outputFile << setprecision(precision) << data[row] << endl;
        }
        outputFile.close();
    }


    thrust::host_vector<double> read1DFile(string fileName, long numRows) {
        thrust::host_vector<double> data;
        this->openInputFile(fileName);
        string inputString;
        double tmp;
        for (long row = 0; row < numRows; row++) {
        getline(inputFile, inputString);
        sscanf(inputString.c_str(), "%lf", &tmp);
        data.push_back(tmp);
        }
        inputFile.close();
        return data;
    }

    /**
     * @brief Save a 1D (double) vector to a file
     * 
     * @param fileName file name
     * @param data double vector to save
     */
    void save1DFile(string fileName, thrust::host_vector<double> data) {
        this->openOutputFile(fileName);
        long numRows = data.size();
        for (long row = 0; row < numRows; row++) {
        //sprintf(outputFile, "%lf \n", data[row]);
        outputFile << setprecision(precision) << data[row] << endl;
        }
        outputFile.close();
    }

    thrust::host_vector<double> read2DFile(string fileName, long numRows) {
        thrust::host_vector<double> data;
        this->openInputFile(fileName);
        string inputString;
        double data1, data2;
        for (long row = 0; row < numRows; row++) {
        getline(inputFile, inputString);
        sscanf(inputString.c_str(), "%lf %lf", &data1, &data2);
        data.push_back(data1);
        data.push_back(data2);
        }
        inputFile.close();
        return data;
    }

    void save2DFile(string fileName, thrust::host_vector<double> data, long numCols) {
        this->openOutputFile(fileName);
        long numRows = int(data.size()/numCols);
        for (long row = 0; row < numRows; row++) {
        for(long col = 0; col < numCols; col++) {
            outputFile << setprecision(precision) << data[row * numCols + col] << "\t";
        }
        outputFile << endl;
        }
        outputFile.close();
    }

    void saveParticlePacking(string dirName) {
        // save scalars
        string fileParams = dirName + "params.dat";
        ofstream saveParams(fileParams.c_str());
        openOutputFile(fileParams);
        saveParams << "numParticles" << "\t" << dpm_->getNumParticles() << endl;
        saveParams << "dt" << "\t" << dpm_->dt << endl;
        saveParams << "phi" << "\t" << dpm_->getParticlePhi() << endl;
        saveParams << "energy" << "\t" << dpm_->getParticleEnergy() / dpm_->getNumParticles() << endl;
        saveParams << "temperature" << "\t" << dpm_->getParticleTemperature() << endl;
        saveParams.close();
        // save vectors
        save1DFile(dirName + "boxSize.dat", dpm_->getBoxSize());
        save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
        save1DFile(dirName + "particleRad.dat", dpm_->getParticleRadii());
    }

    void readBoxSize(string dirName, long nDim_) {
        thrust::host_vector<double> boxSize_(nDim_);
        boxSize_ = read1DFile(dirName + "boxSize.dat", nDim_);
        dpm_->setBoxSize(boxSize_);
    }

    void readParticlePositions(string dirName, long numParticles_, long nDim_) {
        thrust::host_vector<double> particlePos_(numParticles_ * nDim_);
        particlePos_ = read2DFile(dirName + "particlePos.dat", numParticles_);
        dpm_->setParticlePositions(particlePos_);
    }

    void readParticleRadii(string dirName, long numParticles_) {
        thrust::host_vector<double> particleRad_(numParticles_);
        particleRad_ = read1DFile(dirName + "particleRadii.dat", numParticles_);
        dpm_->setParticleRadii(particleRad_);
    }

    void readParticleVelocities(string dirName, long numParticles_, long nDim_) {
        thrust::host_vector<double> particleVel_(numParticles_ * nDim_);
        particleVel_ = read2DFile(dirName + "particleVel.dat", numParticles_);
        dpm_->setParticleVelocities(particleVel_);
    }

    bool containsSubstring(const std::string& inputString, const std::vector<std::string>& substrings) {
        for (const auto& substring : substrings) {
            if (inputString.find(substring) != std::string::npos) {
                return true;  // Substring found
            }
        }
        return false;  // No substrings found
    }

    std::string getLastState(std::string dirName) {
        std::string last_state = "";
        long max_step = -1;
        for (const auto & entry : std::experimental::filesystem::directory_iterator(dirName)) {
            std::string current_file = entry.path().filename().string();
            if (current_file.find("t") == 0) {
                long step = std::stol(current_file.substr(1));
                if (step > max_step) {
                    max_step = step;
                    last_state = entry.path().string() + "/";
                }
            }
        }
        std::cout << "FileIO::getLastState: " << last_state << std::endl;
        return last_state;
    }

    long getLastStep(std::string dirName) {
        long max_step = -1;
        for (const auto & entry : std::experimental::filesystem::directory_iterator(dirName)) {
            std::string current_file = entry.path().filename().string();
            if (current_file.find("t") == 0) {
                long step = std::stol(current_file.substr(1));
                if (step > max_step) {
                    max_step = step;
                }
            }
        }
        return max_step;
    }

    void readPackingFromDirectory(string dirName, long nDim_, bool rigid = false, bool rotation = true) {
        std::cout << "FileIO::readPackingFromDirectory: reading packing from directory: " << dirName << std::endl;
        // check if the directory contains a "system" directory
        std::string system_dir = dirName + "system/";
        if (std::experimental::filesystem::exists(system_dir)) {
            std::cout << "FileIO::readPackingFromDirectory: system directory exists: " << system_dir << std::endl;
        } else {
            std::cout << "FileIO::readPackingFromDirectory: system directory does not exist: " << system_dir << std::endl;
            system_dir = dirName;
        }
        std::string trajectory_dir = dirName + "trajectories/";
        if (std::experimental::filesystem::exists(trajectory_dir)) {
            std::cout << "FileIO::readPackingFromDirectory: trajectory directory exists: " << trajectory_dir << std::endl;
        } else {
            std::cout << "FileIO::readPackingFromDirectory: trajectory directory does not exist: " << trajectory_dir << std::endl;
            trajectory_dir = system_dir;
        }

        std::cout << "FileIO::readPackingFromDirectory: reading energies" << std::endl;
        
        // read the energies
        double ea = readDoubleValueFromParams(system_dir + "params.dat", "ea");
        double el = readDoubleValueFromParams(system_dir + "params.dat", "el");
        double eb = readDoubleValueFromParams(system_dir + "params.dat", "eb");
        double ec = readDoubleValueFromParams(system_dir + "params.dat", "ec");
        dpm_->setEnergyCosts(ea, el, eb, ec);

        std::cout << "FileIO::readPackingFromDirectory: reading thermostat variables" << std::endl;
        // read the thermostat variables
        dpm_->T0 = readDoubleValueFromParams(system_dir + "params.dat", "targetTemperature");
        dpm_->eta = readDoubleValueFromParams(system_dir + "params.dat", "eta");
        dpm_->pi = readDoubleValueFromParams(system_dir + "params.dat", "pi");
        dpm_->Q_thermostat = readDoubleValueFromParams(system_dir + "params.dat", "Q_thermostat");

        std::cout << "FileIO::readPackingFromDirectory: reading particle rigidity" << std::endl;
        // get the particle rigidity from the directory name - one of "rigid", "soft"
        if (containsSubstring(dirName, {"rigid"})) {
            dpm_->setDeformability(simControlStruct::deformabilityEnum::rigid);
            rigid = true;
        } else if (containsSubstring(dirName, {"soft"}) || containsSubstring(dirName, {"deformable"})) {
            dpm_->setDeformability(simControlStruct::deformabilityEnum::deformable);
        } else {
            cerr << "ioDPMFile::readPackingFromDirectory: error: could not determine particle rigidity from directory name " << dirName << endl;
            exit(1);
        }

        // get the particle type from the directory name - one of "disk", "dimer", "dpm"
        if (containsSubstring(dirName, {"disk"})) {
            dpm_->setParticleType(simControlStruct::particleTypeEnum::disk);
            std::cout << "FileIO::readPackingFromDirectory: particle type is disk" << std::endl;
            std::cout << "FileIO::readPackingFromDirectory: particle type is disk" << std::endl;
            std::cout << "FileIO::readPackingFromDirectory: particle type is disk" << std::endl;
            std::cout << "FileIO::readPackingFromDirectory: particle type is disk" << std::endl;
            std::cout << "FileIO::readPackingFromDirectory: particle type is disk" << std::endl;
        } else if (containsSubstring(dirName, {"dimer"})) {
            dpm_->setParticleType(simControlStruct::particleTypeEnum::dimer);
            std::cout << "FileIO::readPackingFromDirectory: particle type is dimer" << std::endl;
        } else if (containsSubstring(dirName, {"dpm"})) {
            dpm_->setParticleType(simControlStruct::particleTypeEnum::dpm);
            std::cout << "FileIO::readPackingFromDirectory: particle type is dpm" << std::endl;
            std::cout << "FileIO::readPackingFromDirectory: particle type is dpm" << std::endl;
            std::cout << "FileIO::readPackingFromDirectory: particle type is dpm" << std::endl;
            std::cout << "FileIO::readPackingFromDirectory: particle type is dpm" << std::endl;
            std::cout << "FileIO::readPackingFromDirectory: particle type is dpm" << std::endl;
        } else {
            cerr << "ioDPMFile::readPackingFromDirectory: error: could not determine particle type from directory name " << dirName << endl;
            exit(1);
        }

        std::cout << "FileIO::readPackingFromDirectory: reading interaction type" << std::endl;
        // get the interaction type from the directory name - one of "smooth", "bumpy"
        if (containsSubstring(dirName, {"smooth"})) {
            dpm_->setSmoothness(simControlStruct::smoothnessEnum::smooth);
        } else if (containsSubstring(dirName, {"bumpy"})) {
            dpm_->setSmoothness(simControlStruct::smoothnessEnum::bumpy);
        } else {
            cerr << "ioDPMFile::readPackingFromDirectory: error: could not determine interaction type from directory name " << dirName << endl;
            exit(1);
        }

        std::cout << "FileIO::readPackingFromDirectory: reading potential type" << std::endl;
        // get the potential type from the directory name - one of "wca", "harmonic"
        if (containsSubstring(dirName, {"wca"})) {
            dpm_->setPotentialType(simControlStruct::potentialEnum::wca);

        } else if (containsSubstring(dirName, {"harmonic"})) {
            dpm_->setPotentialType(simControlStruct::potentialEnum::harmonic);
        } else if (containsSubstring(dirName, {"hertzian"})) {
            dpm_->setPotentialType(simControlStruct::potentialEnum::hertzian);
        } else {
            cerr << "ioDPMFile::readPackingFromDirectory: error: could not determine potential type from directory name " << dirName << endl;
            exit(1);
        }

        std::cout << "FileIO::readPackingFromDirectory: reading numParticles and numVertexPerParticle" << std::endl;
        long numParticles_ = readDoubleValueFromParams(system_dir + "params.dat", "numParticles");
        dpm_->setNumParticles(numParticles_);
        long numVertexPerParticle_ = readDoubleValueFromParams(system_dir + "params.dat", "numVertexPerParticle");
        dpm_->setNumVertexPerParticle(numVertexPerParticle_);
        thrust::host_vector<long> numVertexInParticleList_(numParticles_);

        std::cout << "FileIO::readPackingFromDirectory: reading numVertexInParticleList" << std::endl;
        numVertexInParticleList_ = read1DIndexFile(system_dir + "numVertexInParticleList.dat", numParticles_);
        dpm_->setNumVertexInParticleList(numVertexInParticleList_);
        long numVertices_ = thrust::reduce(numVertexInParticleList_.begin(), numVertexInParticleList_.end(), 0, thrust::plus<long>());
        dpm_->setNumVertices(numVertices_);
        cout << "FileIO::readPackingFromDirectory: numVertices: " << numVertices_ << " on device: " << dpm_->getNumVertices() << endl;

        std::cout << "FileIO::readPackingFromDirectory: resizing firstVertexInParticleList" << std::endl;
        dpm_->resizeFirstVertexInParticleList(numParticles_);
        // i think this is causing some issues:
        // dpm_->setMonoSizeDistribution();
        // ^^^^^^^^^^^^^^^^^^^^^^^^^

        std::cout << "FileIO::readPackingFromDirectory: initializing particleIdList" << std::endl;
        dpm_->initParticleIdList();

        std::cout << "FileIO::readPackingFromDirectory: initializing particle variables" << std::endl;

		dpm_->initParticleVariables(numParticles_);

        std::cout << "FileIO::readPackingFromDirectory: initializing shape variables" << std::endl;
        dpm_->initShapeVariables(numVertices_, numParticles_);

        std::cout << "FileIO::readPackingFromDirectory: initializing dynamical variables" << std::endl;
        dpm_->initDynamicalVariables(numVertices_);

        std::cout << "FileIO::readPackingFromDirectory: initializing neighbors" << std::endl;
        dpm_->initNeighbors(numVertices_);

        std::cout << "FileIO::readPackingFromDirectory: initializing contacts" << std::endl;
        dpm_->initContacts(numParticles_);

        std::cout << "FileIO::readPackingFromDirectory: syncing neighbors to device" << std::endl;
        dpm_->syncNeighborsToDevice();

        std::cout << "FileIO::readPackingFromDirectory: initializing particle neighbors" << std::endl;
		dpm_->initParticleNeighbors(numParticles_);

        std::cout << "FileIO::readPackingFromDirectory: syncing particle neighbors to device" << std::endl;
		dpm_->syncParticleNeighborsToDevice();

        thrust::host_vector<double> boxSize_(nDim_);
        thrust::host_vector<double> pos_(numVertices_ * nDim_);
        thrust::host_vector<double> vel_(numVertices_ * nDim_);
        thrust::host_vector<double> force_(numVertices_ * nDim_);
        thrust::host_vector<double> rad_(numVertices_);
        thrust::host_vector<double> a0_(numParticles_);
        thrust::host_vector<double> l0_(numVertices_);
        thrust::host_vector<double> theta0_(numVertices_);
        thrust::host_vector<double> vertexMass_(numVertices_);

        thrust::host_vector<double> particleRadii_(numParticles_);
        particleRadii_ = read1DFile(system_dir + "particleRadii.dat", numParticles_);
        dpm_->setParticleRadii(particleRadii_);

        std::string last_state_dirName = getLastState(trajectory_dir);
        if (last_state_dirName == "") {
            last_state_dirName = system_dir;
        }

        boxSize_ = read1DFile(system_dir + "boxSize.dat", nDim_);
        dpm_->setBoxSize(boxSize_);
        pos_ = read2DFile(last_state_dirName + "positions.dat", numVertices_);
        dpm_->setVertexPositions(pos_);

        // If last_state_dirName + "velocities.dat" exists, read it
        if (std::experimental::filesystem::exists(last_state_dirName + "velocities.dat")) {
            vel_ = read2DFile(last_state_dirName + "velocities.dat", numVertices_);
            dpm_->setVertexVelocities(vel_);
        } else if (rigid) {
            std::cout << "Warning: " << last_state_dirName << "velocities.dat does not exist." << std::endl;
        } else {
            // throw an error
            std::cerr << "Error: " << last_state_dirName << "velocities.dat does not exist." << std::endl;
            exit(1);
        }

        if (std::experimental::filesystem::exists(last_state_dirName + "forces.dat")) {
            force_ = read2DFile(last_state_dirName + "forces.dat", numVertices_);
            dpm_->setVertexForces(force_);
        } else {
            std::cout << "Warning: " << last_state_dirName << "forces.dat does not exist." << std::endl;
        }

        rad_ = read1DFile(system_dir + "radii.dat", numVertices_);
        dpm_->setVertexRadii(rad_);
        a0_ = read1DFile(system_dir + "restAreas.dat", numParticles_);
        dpm_->setRestAreas(a0_);
        l0_ = read1DFile(system_dir + "restLengths.dat", numVertices_);
        dpm_->setRestLengths(l0_);
        theta0_ = read1DFile(system_dir + "restAngles.dat", numVertices_);
        dpm_->setRestAngles(theta0_);
        vertexMass_ = read1DFile(system_dir + "vertexMasses.dat", numVertices_);
        dpm_->setVertexMasses(vertexMass_);

        // TODO handle this better
        if (rigid) {

            dpm_->initRotationalVariables(numVertices_, numParticles_);

            if (rotation) {
                if (std::experimental::filesystem::exists(last_state_dirName + "particleAngVel.dat")) {
                    thrust::host_vector<double> particleAngVel_(numParticles_);
                    particleAngVel_ = read1DFile(last_state_dirName + "particleAngVel.dat", numParticles_);
                    dpm_->setParticleAngularVelocities(particleAngVel_);
                } else {
                    std::cout << "Warning: " << last_state_dirName << "particleAngVel.dat does not exist." << std::endl;
                }

                if (std::experimental::filesystem::exists(last_state_dirName + "particleTorques.dat")) {
                    thrust::host_vector<double> particleTorque_(numParticles_);
                    particleTorque_ = read1DFile(last_state_dirName + "particleTorques.dat", numParticles_);
                    dpm_->setParticleTorques(particleTorque_);
                } else {
                    std::cout << "Warning: " << last_state_dirName << "particleTorques.dat does not exist." << std::endl;
                }

            }

            if (std::experimental::filesystem::exists(last_state_dirName + "particleAngles.dat")) {
                thrust::host_vector<double> particleAngle_(numParticles_);
                particleAngle_ = read1DFile(last_state_dirName + "particleAngles.dat", numParticles_);
                dpm_->setParticleAngles(particleAngle_);
            } else {
                std::cout << "Warning: " << last_state_dirName << "particleAngles.dat does not exist." << std::endl;
            }

            if (std::experimental::filesystem::exists(last_state_dirName + "particleVelocities.dat")) {
                thrust::host_vector<double> particleVel_(numParticles_ * nDim_);
                particleVel_ = read2DFile(last_state_dirName + "particleVelocities.dat", numParticles_);
                dpm_->setParticleVelocities(particleVel_);
            } else {
                std::cout << "Warning: " << last_state_dirName << "particleVelocities.dat does not exist." << std::endl;
            }

            thrust::host_vector<double> particlePos_(numParticles_ * nDim_);
            particlePos_ = read2DFile(last_state_dirName + "particlePos.dat", numParticles_);
            dpm_->setParticlePositions(particlePos_);

            if (std::experimental::filesystem::exists(last_state_dirName + "particleForces.dat")) {
                thrust::host_vector<double> particleForce_(numParticles_ * nDim_);
                particleForce_ = read2DFile(last_state_dirName + "particleForces.dat", numParticles_);
                dpm_->setParticleForces(particleForce_);
            } else {
                std::cout << "Warning: " << last_state_dirName << "particleForces.dat does not exist." << std::endl;
            }
        }

        // set length scales
        double particle_sigma = dpm_->getMeanParticleSigma();
        dpm_->scalePacking(particle_sigma);
        dpm_->setLengthScale();
        cout << "FileIO::readPackingFromDirectory: preferred phi: " << dpm_->getPreferredPhi() << endl;
        dpm_->calcParticlesShape();  // TODO MAKE AN AREA ENUM

        if (rigid) {
            dpm_->calcParticleMomentInertia();
        }

        dpm_->setForceType();

        std::cout << "FileIO::readPackingFromDirectory: Completed Reading From: " << dirName << std::endl;
    }

    double readDoubleValueFromParams(string fileParamsPath, string key, bool exitOnFail = true) {
        std::cout << "FileIO::readDoubleValueFromParams: " << fileParamsPath << " " << key << std::endl;
        std::ifstream fileParams(fileParamsPath);
        std::string line;
        double value = -1.0;
        if (fileParams.is_open()) {
            // Read the file line by line
            while (getline(fileParams, line)) {
                // Use a string stream to separate the key and the value
                std::istringstream iss(line);
                std::string header;
                double temp;
                if (getline(iss, header, '\t')) { // Use tab as the delimiter
                    iss >> temp;
                    // Check if the key is what we're looking for
                    if (header == key) {
                        value = temp;
                        break; // Stop searching once we find the key
                    }
                }
            }
            fileParams.close(); // Close the file
        } else {
            if (exitOnFail) {
                std::cerr << "Unable to open file" << fileParamsPath << std::endl;
            } else {
                std::cout << "Warning: " << fileParamsPath << " does not exist.  Defaulting to -1." << std::endl;
                value = -1.0;
            }
        }
        std::cout << "FileIO::readDoubleValueFromParams: " << key << " " << value << std::endl;
        return value;
    }

    long readLongValueFromParams(string fileParamsPath, string key, bool exitOnFail = true) {
        std::cout << "FileIO::readLongValueFromParams: " << fileParamsPath << " " << key << std::endl;
        std::ifstream fileParams(fileParamsPath);
        std::string line;
        long value = -1;
        if (fileParams.is_open()) {
            // Read the file line by line
            while (getline(fileParams, line)) {
                // Use a string stream to separate the key and the value
                std::istringstream iss(line);
                std::string header;
                long temp;
                if (getline(iss, header, '\t')) { // Use tab as the delimiter
                    iss >> temp;
                    // Check if the key is what we're looking for
                    if (header == key) {
                        value = temp;
                        break; // Stop searching once we find the key
                    }
                }
            }
            fileParams.close(); // Close the file
        } else {
            if (exitOnFail) {
                std::cerr << "Unable to open file" << fileParamsPath << std::endl;
            } else {
                std::cout << "Warning: " << fileParamsPath << " does not exist.  Defaulting to -1." << std::endl;
                value = -1;
            }
        }
        return value;
    }

    /**
     * @brief Handles the saving of integrator states, console data, and energy data at specified intervals.
     * 
     * @param outDir Directory where the output files will be saved.
     * @param step Current simulation step.
     * @param min_save_decade Minimum frequency for saving states in terms of steps.
     * @param multiple Reference to a variable that tracks the current multiple of config_freq.
     * @param saveFreq Reference to a variable that determines the current save frequency.
     * @param console_freq Frequency (in steps) at which console data should be written.
     * @param config_freq Frequency (in steps) at which configuration states should be saved.
     * @param energy_freq Frequency (in steps) at which energy data should be saved.
     * @param save_log If true, saves states in a logarithmic fashion; otherwise, saves states at regular intervals.
     * @param force_save If true, forces saving regardless of the current step.
     */
    void handleIntegratorSaving(string outDir, long step, long min_save_decade, long& multiple, long& saveFreq, long console_freq, long config_freq, long energy_freq, bool save_log = true, bool force_save = false) {
        if (outDir != "") {
            if (save_log) {
                if (step > multiple * config_freq || force_save) {
                    saveFreq = min_save_decade;
                    multiple += 1;
                }
                if ((step - (multiple - 1) * config_freq) > saveFreq * 10 || force_save) {
                    saveFreq *= 10;
                }
                if ((step - (multiple - 1) * config_freq) % saveFreq == 0) {
                    string current_dir = outDir + "trajectories/t" + std::to_string(step) + "/";
                    makeDir(current_dir, false);
                    saveStates(current_dir);
                }
            } else {
                if (step % config_freq == 0 || force_save) {
                    string current_dir = outDir + "trajectories/t" + std::to_string(step) + "/";
                    makeDir(current_dir, false);
                    saveStates(current_dir);
                }
            }
            
            if (step % energy_freq == 0 || force_save) {
                saveEnergies(step, dpm_->dt);
            }
        }
        if (step % console_freq == 0) {
            writeConsoleData(step);
        }
    }

    template <typename T>
    void writeKeyValToParams(string dirName, string key, T value) {
        string fileParams = dirName + "params.dat";
        ofstream saveParams(fileParams.c_str(), ios::app);  // Open in append mode
        if (!saveParams.is_open()) {
            cerr << "Error opening file: " << fileParams << endl;
            return;
        }
        saveParams << key << "\t" << value << endl;
        saveParams.close();
    }

    void savePacking(string dirName, bool rigid = false) {
        // save scalars
        string fileParams = dirName + "params.dat";
        ofstream saveParams(fileParams.c_str());
        openOutputFile(fileParams);
        saveParams << "numParticles" << "\t" << dpm_->getNumParticles() << endl;
        saveParams << "numVertexPerParticle" << "\t" << dpm_->getNumVertexPerParticle() << endl;
        saveParams << "ea" << "\t" << dpm_->ea << endl;
        saveParams << "el" << "\t" << dpm_->el << endl;
        saveParams << "eb" << "\t" << dpm_->eb << endl;
        saveParams << "ec" << "\t" << dpm_->ec << endl;
        saveParams << "dt" << "\t" << dpm_->dt << endl;
        saveParams << "phi" << "\t" << dpm_->getPhi() << endl;
        saveParams << "phi0" << "\t" << dpm_->getPreferredPhi() << endl;
        saveParams << "epot" << "\t" << dpm_->getPotentialEnergy() / dpm_->getNumParticles() << endl;
        if (rigid) {
            saveParams << "temperature" << "\t" << dpm_->getRigidTemperature() << endl;
        }
        else {
            saveParams << "temperature" << "\t" << dpm_->getTemperature() << endl;
        }
        saveParams << "targetTemperature" << "\t" << dpm_->T0 << endl;
        saveParams << "eta" << "\t" << dpm_->eta << endl;
        saveParams << "pi" << "\t" << dpm_->pi << endl;
        saveParams << "Q_thermostat" << "\t" << dpm_->Q_thermostat << endl;
        saveParams << "nDim" << "\t" << dpm_->nDim << endl;
        saveParams << "segmentLengthPerVertexRad" << "\t" << dpm_->segmentLengthPerVertexRad << endl;
        saveParams.close();
        // save vectors
        save1DFile(dirName + "boxSize.dat", dpm_->getBoxSize());
        save1DIndexFile(dirName + "numVertexInParticleList.dat", dpm_->getNumVertexInParticleList());
        save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
        save1DFile(dirName + "radii.dat", dpm_->getVertexRadii());
        save1DFile(dirName + "restAreas.dat", dpm_->getRestAreas());
        save1DFile(dirName + "restLengths.dat", dpm_->getRestLengths());
        save1DFile(dirName + "restAngles.dat", dpm_->getRestAngles());
        save2DFile(dirName + "velocities.dat", dpm_->getVertexVelocities(), dpm_->nDim);
        save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
        save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
        save1DFile(dirName + "vertexMasses.dat", dpm_->getVertexMasses());
        save1DFile(dirName + "particleRadii.dat", dpm_->getParticleRadii());

        if (rigid) {
            save1DFile(dirName + "particleTorques.dat", dpm_->getParticleTorques());
            save1DFile(dirName + "particleAngVel.dat", dpm_->getParticleAngularVelocities());
            save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
            save2DFile(dirName + "particleVelocities.dat", dpm_->getParticleVelocities(), dpm_->nDim);
            save2DFile(dirName + "particleForces.dat", dpm_->getParticleForces(), dpm_->nDim);
        }
    }

    void readVertexVelocities(string dirName, long numVertices_, long nDim_) {
        thrust::host_vector<double> vel_(numVertices_ * nDim_);
        vel_ = read2DFile(dirName + "velocities.dat", numVertices_);
        dpm_->setVertexVelocities(vel_);
        // TODO: set the T0 value in the integrator?
    }

    void readState(string dirName, long numParticles_, long numVertices_, long nDim_) {
        thrust::host_vector<double> vel_(numVertices_ * nDim_);
        thrust::host_vector<double> particlePos_(numParticles_ * nDim_);
        thrust::host_vector<double> particleAngle_(numParticles_);
        vel_ = read2DFile(dirName + "velocities.dat", numVertices_);
        dpm_->setVertexVelocities(vel_);
        particlePos_ = read2DFile(dirName + "particlePos.dat", numParticles_);
        dpm_->setParticlePositions(particlePos_);
        particleAngle_ = read1DFile(dirName + "particleAngles.dat", numParticles_);
        dpm_->setParticleAngles(particleAngle_);
    }

    void saveStateByName(const std::string& dirName, const std::string& stateName) {
        if (stateName == "positions") {
            save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
        } else if (stateName == "forces") {
            save2DFile(dirName + "forces.dat", dpm_->getVertexForces(), dpm_->nDim);
        } else if (stateName == "vertexPotential") {
            save1DFile(dirName + "vertexPotential.dat", dpm_->getVertexPotentialEnergies());
        } else if (stateName == "velocities") {
            save2DFile(dirName + "velocities.dat", dpm_->getVertexVelocities(), dpm_->nDim);
        } else if (stateName == "particlePos") {
            save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
        } else if (stateName == "particleAngles") {
            save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
        } else if (stateName == "angles") {
            save1DFile(dirName + "angles.dat", dpm_->getVertexAngles());
        } else if (stateName == "areas") {
            save1DFile(dirName + "areas.dat", dpm_->getAreas());
        } else if (stateName == "lengths") {
            save1DFile(dirName + "lengths.dat", dpm_->getSegmentLengths());
        } else if (stateName == "stressTensor") {
            save1DFile(dirName + "stressTensor.dat", dpm_->getStressTensor());
        } else if (stateName == "particleForces") {
            save2DFile(dirName + "particleForces.dat", dpm_->getParticleForces(), dpm_->nDim);
        } else if (stateName == "particleTorques") {
            save1DFile(dirName + "particleTorques.dat", dpm_->getParticleTorques());
        } else if (stateName == "particleAngVel") {
            save1DFile(dirName + "particleAngVel.dat", dpm_->getParticleAngularVelocities());
        } else if (stateName == "particleVelocities") {
            save2DFile(dirName + "particleVelocities.dat", dpm_->getParticleVelocities(), dpm_->nDim);
        } else if (stateName == "frictionCoefficient") {
            save1DFile(dirName + "frictionCoefficient.dat", dpm_->getFrictionCoefficients());
        } else if (stateName == "boxSize") {
            save2DFile(dirName + "boxSize.dat", dpm_->getBoxSize(), dpm_->nDim);
        } else {
            std::cerr << "FileIO::saveState " << stateName << " not recognized" << std::endl;
        }
    }

    void saveStates(string dirName) {
        for (const auto& stateName : stateNames) {
            saveStateByName(dirName, stateName);
        }
    }

    void writeEnergyFileHeader() {
        energyFile << "step" << "\t" << "time" << "\t";
        for (const auto& energyName : energyNames) {
            energyFile << energyName << "\t";
        }
        energyFile << endl;
    }

    void addValueToEnergyFile(double value) {
        energyFile << setprecision(precision) << value << "\t";
    }

    void saveEnergyByName(const std::string& energyName) {
        if (energyName == "potential_energy_area") {
            addValueToEnergyFile(dpm_->getAreaPotentialEnergy());
        } else if (energyName == "potential_energy_length") {
            addValueToEnergyFile(dpm_->getLengthPotentialEnergy());
        } else if (energyName == "potential_energy_bending") {
            addValueToEnergyFile(dpm_->getBendingPotentialEnergy());
        } else if (energyName == "potential_energy_interaction") {
            addValueToEnergyFile(dpm_->getInteractionPotentialEnergy());
        } else if (energyName == "potential_energy") {
            addValueToEnergyFile(dpm_->getPotentialEnergy());
        } else if (energyName == "kinetic_energy") {
            addValueToEnergyFile(dpm_->getKineticEnergy());
        } else if (energyName == "temperature") {
            addValueToEnergyFile(dpm_->getTemperature());
        } else if (energyName == "total_energy") {
            addValueToEnergyFile(dpm_->getPotentialEnergy() + dpm_->getKineticEnergy());
        } else if (energyName == "potential_energy_rigid") {
            addValueToEnergyFile(dpm_->getParticleEnergy());
        } else if (energyName == "kinetic_energy_rigid") {
            addValueToEnergyFile(dpm_->getRigidKineticEnergy());
        } else if (energyName == "temperature_rigid") {
            addValueToEnergyFile(dpm_->getRigidTemperature());
        } else if (energyName == "total_energy_rigid") {
            addValueToEnergyFile(dpm_->getParticleEnergy() + dpm_->getRigidKineticEnergy());

        } else if (energyName == "temperature_rigid_no_rot") {
            addValueToEnergyFile(dpm_->getRigidTemperature_noRotation());

        } else if (energyName == "phi") {
            addValueToEnergyFile(dpm_->getPhi());
        } else if (energyName == "eta") {
            addValueToEnergyFile(dpm_->eta);
        } else if (energyName == "pi") {
            addValueToEnergyFile(dpm_->pi);
        } else if (energyName == "target_temperature") {
            addValueToEnergyFile(dpm_->T0);
        } else if (energyName == "nose_hoover_extended_energy") {
            addValueToEnergyFile(dpm_->getNoseHooverExtendedEnergy(true));
        } else if (energyName == "nose_hoover_extended_energy_no_rot") {
            addValueToEnergyFile(dpm_->getNoseHooverExtendedEnergy(false));

        } else if (energyName == "mean_friction_coefficient") {
            addValueToEnergyFile(dpm_->getMeanFrictionCoefficient());
        } else if (energyName == "total_vertex_contacts") {
            // addValueToEnergyFile(dpm_->getTotalVertexContacts());
            addValueToEnergyFile(dpm_->getTotalVertexContactsNewNew());
        } else if (energyName == "total_particle_contacts") {
            // addValueToEnergyFile(dpm_->getTotalParticleContacts());
            addValueToEnergyFile(dpm_->getTotalParticleContactsNew());
        } else if (energyName == "total_1_vertex_particle_contacts") {
            // addValueToEnergyFile(dpm_->getTotalParticleContacts());
            const long contactLimit = 1;
            addValueToEnergyFile(dpm_->getTotalParticleContactsByType(contactLimit));
        } else if (energyName == "total_2_vertex_particle_contacts") {
            // addValueToEnergyFile(dpm_->getTotalParticleContacts());
            const long contactLimit = 2;
            addValueToEnergyFile(dpm_->getTotalParticleContactsByType(contactLimit));
        } else if (energyName == "total_3_vertex_particle_contacts") {
            // addValueToEnergyFile(dpm_->getTotalParticleContacts());
            const long contactLimit = 3;
            addValueToEnergyFile(dpm_->getTotalParticleContactsByType(contactLimit));
        } else if (energyName == "total_4_vertex_particle_contacts") {
            // addValueToEnergyFile(dpm_->getTotalParticleContacts());
            const long contactLimit = 4;
            addValueToEnergyFile(dpm_->getTotalParticleContactsByType(contactLimit));
        } else if (energyName == "pressure") {
            addValueToEnergyFile(dpm_->getPressure());
        } else if (energyName == "vertex_overlaps") {
            addValueToEnergyFile(dpm_->getVertexOverlaps());
            // addValueToEnergyFile(dpm_->getTotalVertexOverlapAreasNew());
        } else if (energyName == "particle_overlaps") {
            addValueToEnergyFile(dpm_->getParticleOverlaps());
        } else {
            std::cerr << "FileIO::saveEnergy " << energyName << " not recognized" << std::endl;
        }
    }

    void saveEnergies(long step, double dt) {
        energyFile << step << "\t" << step * dt << "\t";
        for (const auto& energyName : energyNames) {
            saveEnergyByName(energyName);
        }
        energyFile << endl;
    }

    void saveState(string dirName, bool rigid = false) {
        save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
        save2DFile(dirName + "forces.dat", dpm_->getVertexForces(), dpm_->nDim);
        save2DFile(dirName + "velocities.dat", dpm_->getVertexVelocities(), dpm_->nDim);
        save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
        save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
        save1DFile(dirName + "angles.dat", dpm_->getVertexAngles());
        save1DFile(dirName + "areas.dat", dpm_->getAreas());
        save1DFile(dirName + "lengths.dat", dpm_->getSegmentLengths());
        
        // dpm_->checkParticleMaxDisplacement();  // consider moving this into the stress tensor calculation - it is needed to update the particle neighbor list
        save1DFile(dirName + "stressTensor.dat", dpm_->getStressTensor());

        if (rigid) {
            save2DFile(dirName + "particleForces.dat", dpm_->getParticleForces(), dpm_->nDim);
            save1DFile(dirName + "particleTorques.dat", dpm_->getParticleTorques());
            save1DFile(dirName + "particleAngVel.dat", dpm_->getParticleAngularVelocities());
            save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
            save2DFile(dirName + "particleVelocities.dat", dpm_->getParticleVelocities(), dpm_->nDim);
        }
    }

    void saveParticleState(string dirName) {
        save2DFile(dirName + "positions.dat", dpm_->getParticlePositions(), dpm_->nDim);
        save2DFile(dirName + "velocities.dat", dpm_->getParticleVelocities(), dpm_->nDim);
        save2DFile(dirName + "forces.dat", dpm_->getParticleForces(), dpm_->nDim);
    }

    void saveParticleRadii(string dirName) {
        save1DFile(dirName + "particleRadii.dat", dpm_->getParticleRadii());
    }

    void saveParticleVelocities(string dirName) {
        save2DFile(dirName + "particleVel.dat", dpm_->getParticleVelocities(), dpm_->nDim);
    }

    void saveContacts(string dirName) {
        dpm_->calcContacts(0);
        save2DFile(dirName + "contacts.dat", dpm_->getContacts(), dpm_->contactLimit);
    }

    void saveConfiguration(string dirName) {
        savePacking(dirName);
        saveContacts(dirName);
    }

    void readRigidPackingFromDirectory(string dirName, long numParticles_, long nDim_) {
        thrust::host_vector<long> numVertexInParticleList_(numParticles_);
        numVertexInParticleList_ = read1DIndexFile(dirName + "numVertexInParticleList.dat", numParticles_);
        dpm_->setNumVertexInParticleList(numVertexInParticleList_);
        long numVertices_ = thrust::reduce(numVertexInParticleList_.begin(), numVertexInParticleList_.end(), 0, thrust::plus<long>());
        dpm_->setNumVertices(numVertices_);
        cout << "readRigidPackingFromDirectory:: numVertices: " << numVertices_ << " on device: " << dpm_->getNumVertices() << endl;
        dpm_->initParticleIdList();
        dpm_->initShapeVariables(numVertices_, numParticles_);
        dpm_->initDynamicalVariables(numVertices_);
        dpm_->initNeighbors(numVertices_);
        dpm_->initParticleVariables(numParticles_);
        dpm_->initRotationalVariables(numVertices_, numParticles_);
        dpm_->initDeltaVariables(numVertices_, numParticles_);
        thrust::host_vector<double> boxSize_(nDim_);
        thrust::host_vector<double> pos_(numVertices_ * nDim_);
        thrust::host_vector<double> rad_(numVertices_);
        thrust::host_vector<double> a0_(numParticles_);
        thrust::host_vector<double> particlePos_(numParticles_);
        thrust::host_vector<double> particleAngles_(numParticles_);

        boxSize_ = read1DFile(dirName + "boxSize.dat", nDim_);
        dpm_->setBoxSize(boxSize_);
        pos_ = read2DFile(dirName + "positions.dat", numVertices_);
        dpm_->setVertexPositions(pos_);
        rad_ = read1DFile(dirName + "radii.dat", numVertices_);
        dpm_->setVertexRadii(rad_);
        a0_ = read1DFile(dirName + "restAreas.dat", numParticles_);
        dpm_->setRestAreas(a0_);
        particlePos_ = read2DFile(dirName + "particlePos.dat", numParticles_);
        dpm_->setParticlePositions(particlePos_);
        particleAngles_ = read1DFile(dirName + "particleAngles.dat", numParticles_);
        dpm_->setParticleAngles(particleAngles_);
        // set length scales
        dpm_->setLengthScaleToOne();
        cout << "FileIO::readRigidPackingFromDirectory: phi: " << dpm_->getPreferredPhi() << endl;
    }

    void saveRigidPacking(string dirName) {
        // save scalars
        string fileParams = dirName + "params.dat";
        ofstream saveParams(fileParams.c_str());
        openOutputFile(fileParams);
        saveParams << "numParticles" << "\t" << dpm_->getNumParticles() << endl;
        saveParams << "phi" << "\t" << dpm_->getPhi() << endl;
        saveParams << "epot" << "\t" << dpm_->getPotentialEnergy() << endl;
        saveParams << "temperature" << "\t" << dpm_->getRigidTemperature() << endl;
        saveParams.close();
        // save vectors
        save1DFile(dirName + "boxSize.dat", dpm_->getBoxSize());
        save1DIndexFile(dirName + "numVertexInParticleList.dat", dpm_->getNumVertexInParticleList());
        save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
        save1DFile(dirName + "radii.dat", dpm_->getVertexRadii());
        save1DFile(dirName + "restAreas.dat", dpm_->getRestAreas());
        save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
        save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
        save2DFile(dirName + "neighbors.dat", dpm_->getNeighbors(), dpm_->neighborListSize);
    }

    void readRigidState(string dirName, long numParticles_, long nDim_) {;
        thrust::host_vector<double> particleVel_(numParticles_ * nDim_);
        thrust::host_vector<double> particleAngvel_(numParticles_);
        particleVel_ = read2DFile(dirName + "particleVel.dat", numParticles_);
        dpm_->setParticleVelocities(particleVel_);
        particleAngvel_ = read1DFile(dirName + "particleAngvel.dat", numParticles_);
        dpm_->setParticleAngularVelocities(particleAngvel_);

    }

    void saveRigidState(string dirName) {
        save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
        save2DFile(dirName + "velocities.dat", dpm_->getVertexVelocities(), dpm_->nDim);
        save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
        save2DFile(dirName + "particleVel.dat", dpm_->getParticleVelocities(), dpm_->nDim);
        save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
        save1DFile(dirName + "particleAngvel.dat", dpm_->getParticleAngularVelocities());
    }

    void saveSPDPMPacking(string dirName) {
        // save scalars
        string fileParams = dirName + "params.dat";
        ofstream saveParams(fileParams.c_str());
        openOutputFile(fileParams);
        saveParams << "numParticles" << "\t" << dpm_->getNumParticles() << endl;
        saveParams << "ea" << "\t" << dpm_->ea << endl;
        saveParams << "el" << "\t" << dpm_->el << endl;
        saveParams << "eb" << "\t" << dpm_->eb << endl;
        saveParams << "ec" << "\t" << dpm_->ec << endl;
        saveParams << "dt" << "\t" << dpm_->dt << endl;
        saveParams << "phiSP" << "\t" << dpm_->getParticlePhi() << endl;
        saveParams << "phiDPM" << "\t" << dpm_->getPreferredPhi() << endl;
        saveParams << "epot" << "\t" << dpm_->getPotentialEnergy() << endl;
        saveParams << "temperature" << "\t" << dpm_->getTemperature() << endl;
        saveParams.close();
        // save vectors
        save1DFile(dirName + "boxSize.dat", dpm_->getBoxSize());
        save1DIndexFile(dirName + "numVertexInParticleList.dat", dpm_->getNumVertexInParticleList());
        save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
        save1DFile(dirName + "radii.dat", dpm_->getVertexRadii());
        save1DFile(dirName + "restAreas.dat", dpm_->getRestAreas());
        save1DFile(dirName + "restLengths.dat", dpm_->getRestLengths());
        save1DFile(dirName + "restAngles.dat", dpm_->getRestAngles());
        // these two functions are overidden in spdpm2d
        save2DFile(dirName + "softPos.dat", dpm_->getParticlePositions(), dpm_->nDim);
        save1DFile(dirName + "softRad.dat", dpm_->getParticleRadii());
    }

    void saveSPDPMState(string dirName) {
        save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
        save2DFile(dirName + "particleVel.dat", dpm_->getParticleVelocities(), dpm_->nDim);
        save2DFile(dirName + "particleForces.dat", dpm_->getParticleForces(), dpm_->nDim);
    }

    void readFromCellFormat(long skiplines) {// read dpm packing from cell format
        thrust::host_vector<double> boxSize_;
        thrust::host_vector<double> pos_;
        thrust::host_vector<double> rad_;
        thrust::host_vector<double> a0_;
        thrust::host_vector<double> l0_;
        thrust::host_vector<double> theta0_;
        thrust::host_vector<long> numVertexInParticleList_;
        long numParticles_, numVertexInParticle_, numVertices_ = 0;
        double phi_, lx, ly, stress_[MAXDIM + 1];
        double a0tmp, area, p0tmp, xtmp, ytmp, radtmp, l0tmp, theta0tmp, fx, fy;

        string inputString;
        //get rid of first line
        for (long l = 0; l < skiplines; l++) {
        getline(inputFile, inputString);
        }

        // read in simulation information from header
        getline(inputFile, inputString);
        sscanf(inputString.c_str(), "NUMCL %ld", &numParticles_);
        //cout << inputString << "read: " << numParticles_ << endl;

        // verify input file
        if (numParticles_ < 1) {
        cerr << "FileIO::readFromCellFormat: error: numParticles = " << numParticles_ << ". Ending here." << endl;
        exit(1);
        }

        getline(inputFile, inputString);
        sscanf(inputString.c_str(), "PACKF %lf", &phi_);
        //cout << inputString << "read: " << phi_ << endl;

        // initialize box lengths
        getline(inputFile, inputString);
        sscanf(inputString.c_str(), "BOXSZ %lf %lf", &lx, &ly);
        //cout << inputString << "read: " << lx << " " << ly << endl;
        boxSize_.push_back(lx);
        boxSize_.push_back(ly);

        // initialize stress
        getline(inputFile, inputString);
        sscanf(inputString.c_str(), "STRSS %lf %lf %lf", &stress_[0], &stress_[1], &stress_[2]);

        // loop over cells, read in coordinates
        long start = 0;
        for (long particleId = 0; particleId < numParticles_; particleId++) {
        // first parse cell info
        getline(inputFile, inputString);
        sscanf(inputString.c_str(), "CINFO %ld %*d %*d %lf %lf %lf", &numVertexInParticle_, &a0tmp, &area, &p0tmp);
        //cout << "particleId: " << particleId << ", a: " << a0tmp << " " << area << ", p: " << p0tmp << endl;
        numVertexInParticleList_.push_back(numVertexInParticle_);
        numVertices_ += numVertexInParticle_;
        a0_.push_back(a0tmp);

        // loop over vertices and store coordinates
        for (long vertexId = 0; vertexId < numVertexInParticle_; vertexId++) {
            // parse vertex coordinate info
            getline(inputFile, inputString);
            sscanf(inputString.c_str(), "VINFO %*d %*d %lf %lf %lf %lf %lf %lf %lf", &xtmp, &ytmp, &radtmp, &l0tmp, &theta0tmp, &fx, &fy);
            // check pbc
            //xtmp -= floor(xtmp/lx) * lx;
            //ytmp -= floor(ytmp/ly) * ly;
            //cout << "read: vertexId: " << start + vertexId << " forces: " << setprecision(12) << fx << " " << fy << endl;
            //cout << "read: vertexId: " << start + vertexId << " pos: " << setprecision(12) << xtmp << " " << ytmp << endl;
            // push back
            pos_.push_back(xtmp); // push x then y
            pos_.push_back(ytmp);
            rad_.push_back(radtmp);
            l0_.push_back(l0tmp);
            theta0_.push_back(theta0tmp);
        }
        start += numVertexInParticle_;
        }
        inputFile.close();
        // transfer to dpm class
        dpm_->setNumVertices(numVertices_);
        cout << "FileIO::readFromCellFormat: numVertices: " << numVertices_ << " on device: " << dpm_->getNumVertices() << endl;
        // first initilize indexing for polydisperse packing
        dpm_->setNumVertexInParticleList(numVertexInParticleList_);
        dpm_->initParticleIdList();
        dpm_->initShapeVariables(numVertices_, numParticles_);
        dpm_->initDynamicalVariables(numVertices_);
        dpm_->initNeighbors(numVertices_);
        // set all the rest
        dpm_->setBoxSize(boxSize_);
        dpm_->setVertexPositions(pos_);
        dpm_->setVertexRadii(rad_);
        dpm_->setRestAreas(a0_);
        dpm_->setRestLengths(l0_);
        dpm_->setRestAngles(theta0_);
        dpm_->setLengthScale();
        cout << "FileIO::readFromCellFormat: preferred phi: " << dpm_->getPreferredPhi() << endl;
    }
};

#endif // FILEIO_H //