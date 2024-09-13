# CUDA root directory
CUDA_ROOT_DIR = /usr/local/cuda-11.8

# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -Iinclude -I$(CUDA_ROOT_DIR)/include -std=c++17
NVCCFLAGS = -Iinclude -I$(CUDA_ROOT_DIR)/include -std=c++17

# Directories
INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Find all .cpp files that should be treated as executables
EXECUTABLE_SOURCES = $(shell find . -maxdepth 1 -name '*.cpp')
EXECUTABLE_NAMES = $(patsubst ./%.cpp, $(BIN_DIR)/%, $(EXECUTABLE_SOURCES))

# Find all .cu and other .cpp files (not main.cpp) for object generation
CPP_SOURCES = $(shell find $(SRC_DIR) -name '*.cpp')
CUDA_SOURCES = $(shell find $(SRC_DIR) -name '*.cu')

# Object files for non-main source files
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))
CUDA_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CUDA_SOURCES))

# Special rule to compile .cpp files with nvcc if they include CUDA constructs
CUDA_CPP_OBJECTS = $(patsubst %.cpp,$(OBJ_DIR)/%.o, $(wildcard $(SRC_DIR)/*.cpp))

# Default target to build all executables
all: $(EXECUTABLE_NAMES)

# Link each executable from its own main.cpp and other object files
$(BIN_DIR)/%: %.cpp $(CPP_OBJECTS) $(CUDA_OBJECTS) $(CUDA_CPP_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(filter-out $(OBJ_DIR)/main.o,$(CPP_OBJECTS)) $(CUDA_OBJECTS) $(CUDA_CPP_OBJECTS) -o $@

# Compile all .cpp files with g++, except for those including CUDA code
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile all .cu files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Special rule: Compile .cpp files that include CUDA constructs with nvcc
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets
.PHONY: all clean