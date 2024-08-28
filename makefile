# makefile
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := g++

INCLUDES := -Iinclude
CXXFLAGS := -std=c++17
NVCCFLAGS := -arch=sm_50
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart

SRCS := main.cpp src/particles/Particle.cu src/kernels/UtilityKernels.cu
OBJS := bin/main.o bin/particles/Particle.o bin/kernels/UtilityKernels.o

all: main

main: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

bin/main.o: main.cpp | bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

bin/particles/Particle.o: src/particles/Particle.cu | bin/particles
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

bin/kernels/UtilityKernels.o: src/kernels/UtilityKernels.cu | bin/kernels
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

bin:
	mkdir -p bin

bin/particles:
	mkdir -p bin/particles

bin/kernels:
	mkdir -p bin/kernels

clean:
	rm -f $(OBJS) main

.PHONY: clean