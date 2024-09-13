# Makefile

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

# Find all .cpp and .cu files recursively
CPP_SOURCES = $(shell find . -name '*.cpp')
CUDA_SOURCES = $(shell find $(SRC_DIR) -name '*.cu')

# Object files
CPP_OBJECTS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_SOURCES)))
CUDA_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CUDA_SOURCES))

# Target executable
TARGET = $(BIN_DIR)/simulation

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(CPP_OBJECTS) $(CUDA_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $^ -o $@

# Compile .cpp files
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile .cu files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets
.PHONY: all clean
