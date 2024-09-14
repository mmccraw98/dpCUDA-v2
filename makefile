# Compiler and flags
CXX = nvcc
CXXFLAGS = -std=c++14 -O2  # Updated to C++14

# Directories
INCDIR = ./include
SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

# Source files
SRCS = main.cpp $(SRCDIR)/particle.cu $(SRCDIR)/disk.cu

# Object files
OBJS = $(OBJDIR)/main.o $(OBJDIR)/particle.o $(OBJDIR)/disk.o

# Executable
TARGET = $(BINDIR)/md_simulation

# Include directories
INCLUDES = -I$(INCDIR)

# Thrust library (in case it's needed explicitly)
LIBS = 

# Rules
all: $(TARGET)

# Link the object files to create the final executable
$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LIBS)

# Compile the main.cpp file
$(OBJDIR)/main.o: main.cpp $(INCDIR)/disk.h  # Include disk.h dependency
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c main.cpp -o $(OBJDIR)/main.o

# Compile the particle.cu file
$(OBJDIR)/particle.o: $(SRCDIR)/particle.cu $(INCDIR)/particle.h
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(SRCDIR)/particle.cu -o $(OBJDIR)/particle.o

# Compile the disk.cu file
$(OBJDIR)/disk.o: $(SRCDIR)/disk.cu $(INCDIR)/disk.h
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(SRCDIR)/disk.cu -o $(OBJDIR)/disk.o

# Clean up object and binary files
clean:
	rm -rf $(OBJDIR) $(BINDIR)
