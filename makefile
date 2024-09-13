###########################################################

## USER-SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-11.8

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=/usr/bin/g++
CC_FLAGS= -O3 -std=c++17 -I$(CUDA_ROOT_DIR)/include
CC_LIBS= -lstdc++fs

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -O3 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -diag-suppress=550 -Wno-deprecated-gpu-targets
NVCC_LIBS=

LFLAGS= -lm -Wno-deprecated-gpu-targets -fstack-protector 

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src
# Object file directory:
OBJ_DIR = bin
# Executable output directory:
EXEC_DIR = bin/executables
# Include header file directory:
INC_DIR = include

# Specific directories for Particle, Simulator, IO modules, and Kernels
PARTICLE_DIR = $(SRC_DIR)/particle
SIMULATOR_DIR = $(SRC_DIR)/simulator
IO_DIR = $(SRC_DIR)/io
KERNELS_DIR = $(SRC_DIR)/kernels
SCRIPT_DIR = scripts
TEST_DIR = tests

##########################################################

## Make variables ##

# Find all .cpp and .cu files dynamically
SCRIPT_CPP_FILES := $(wildcard $(SCRIPT_DIR)/*.cpp)
TEST_CPP_FILES := $(wildcard $(TEST_DIR)/*.cpp)
SRC_CPP_FILES := $(wildcard $(SRC_DIR)/**/*.cpp)
CU_FILES := $(wildcard $(SRC_DIR)/**/*.cu)

# Convert source files to object files
SCRIPT_OBJ_FILES := $(patsubst $(SCRIPT_DIR)/%.cpp,$(OBJ_DIR)/scripts/%.o,$(SCRIPT_CPP_FILES))
TEST_OBJ_FILES := $(patsubst $(TEST_DIR)/%.cpp,$(OBJ_DIR)/tests/%.o,$(TEST_CPP_FILES))
SRC_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/src/%.o,$(SRC_CPP_FILES))
CU_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/src/%.o,$(CU_FILES))

# All object files
OBJ_FILES := $(SCRIPT_OBJ_FILES) $(TEST_OBJ_FILES) $(SRC_OBJ_FILES) $(CU_OBJ_FILES)

# Generate a list of executables based on files in scripts/ and tests/
EXECUTABLES := $(patsubst $(SCRIPT_DIR)/%.cpp,$(EXEC_DIR)/%,$(SCRIPT_CPP_FILES)) \
               $(patsubst $(TEST_DIR)/%.cpp,$(EXEC_DIR)/%,$(TEST_CPP_FILES))

##########################################################

## Compile ##

# Compiler-specific flags:
GENCODE_SM60 = -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE = $(GENCODE_SM60)

# Generic rule for all executables:
all: $(EXECUTABLES)

# Rule to link object files to create executables in bin/executables/
$(EXECUTABLES): $(EXEC_DIR)/%: $(OBJ_DIR)/scripts/%.o $(CU_OBJ_FILES) $(SRC_OBJ_FILES)
	@mkdir -p $(dir $@)
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $< $(CU_OBJ_FILES) $(SRC_OBJ_FILES) -o $@ $(CC_LIBS)

# Rule for compiling script .cpp files to object files
$(OBJ_DIR)/scripts/%.o: $(SCRIPT_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CC_FLAGS) -I$(INC_DIR) -c $< -o $@

# Rule for compiling test .cpp files to object files
$(OBJ_DIR)/tests/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CC_FLAGS) -I$(INC_DIR) -c $< -o $@

# Rule for compiling source .cpp files to object files
$(OBJ_DIR)/src/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CC_FLAGS) -I$(INC_DIR) -c $< -o $@

# Rule for compiling .cu files to object files (includes kernel files)
$(OBJ_DIR)/src/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) -I$(INC_DIR) -I$(CUDA_ROOT_DIR)/include -c $< -o $@

# Clean objects in object directory and executables
clean:
	rm -f $(OBJ_DIR)/*/*.o $(EXEC_DIR)/*

##########################################################

# Phony targets to prevent naming conflicts with file names
.PHONY: all clean
