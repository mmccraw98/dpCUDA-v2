# Makefile

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -Iinclude
NVCCFLAGS = -std=c++11 -Iinclude
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SOURCES_CPP = $(wildcard $(SRC_DIR)/**/*.cpp) main.cpp
SOURCES_CU = $(wildcard $(SRC_DIR)/**/*.cu)
OBJECTS_CPP = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SOURCES_CPP))
OBJECTS_CU = $(patsubst %.cu, $(OBJ_DIR)/%.o, $(SOURCES_CU))
OBJECTS = $(OBJECTS_CPP) $(OBJECTS_CU)
EXECUTABLE = $(BIN_DIR)/simulation

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(OBJ_DIR)/$(dir $<)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(OBJ_DIR)/$(dir $<)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean