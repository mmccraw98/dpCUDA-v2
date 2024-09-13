###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-11.8

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=/usr/bin/g++
CC_FLAGS= -O3
CC_LIBS= -lstdc++fs

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -O3 -Wno-deprecated-gpu-targets --expt-extended-lambda --expt-relaxed-constexpr -diag-suppress 550
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

# Include header file directory:
INC_DIR = include

##########################################################

## Make variables ##

# Executables:
EXECUTABLES = main

##########################################################

## Compile ##

# Compiler-specific flags:
GENCODE_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM60 = -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE = $(GENCODE_SM60)

# Generic rule for all executables:
all: $(EXECUTABLES)

# Rule to link C++ and CUDA compiled object files to each target executable:
$(EXECUTABLES) : %: $(OBJ_DIR)/%.o $(OBJ_DIR)/DPM2D.o $(OBJ_DIR)/particle.o $(OBJ_DIR)/disk.o
	$(NVCC) $(GENCODE) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $^ -o $@ $(CC_LIBS)

# Compile main .cpp files to object files for each executable:
$(OBJ_DIR)/%.o : %.cpp
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.h
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cuh
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) $(OBJ_DIR)/* $(EXECUTABLES)

##########################################################
