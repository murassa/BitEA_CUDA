SEARCH_PATH = include
CFLAGS = -Wall -std=gnu17 -I$(SEARCH_PATH) -lstdc++
LINKFLAGS = -lm
INCLUDES = -I/usr/local/cuda/include
CUDAFLAGS = -I/usr/local/cuda/include -I$(SEARCH_PATH) -lstdc++
NVCCFLAGS = -arch=sm_50 -std=c++17

ifdef DEBUG
  CFLAGS += -g -D WITH_DEBUG=1
  NVCCFLAGS += -G -D WITH_DEBUG=1
else
  CFLAGS += -O3
  NVCCFLAGS += -O3
endif

# List of source files
SOURCES := $(wildcard src/*.c)
CUDA_SOURCES := $(wildcard src/*.cu)

# Define object files (use different dirs for .c and .cu objects)
OBJECTS := $(SOURCES:src/%.c=obj/c/%.o)
CUDA_OBJECTS := $(CUDA_SOURCES:src/%.cu=obj/cu/%.o)

# Compile C files with gcc
obj/c/%.o: src/%.c
	@mkdir -p $(@D)
	gcc $(CFLAGS) -c $< -o $@ $(INCLUDES)

# Compile CUDA files with nvcc
obj/cu/%.o: src/%.cu
	@mkdir -p $(@D)
	nvcc $(NVCCFLAGS) -c $< -o $@ $(CUDAFLAGS)

# Link all object files to create the executable
build: $(OBJECTS) $(CUDA_OBJECTS)
	g++ $(CFLAGS) $(OBJECTS) $(CUDA_OBJECTS) -o BitEA $(LINKFLAGS) $(INCLUDES) -L/usr/local/cuda/lib64 -lcudart

# Clean up object files and executable
clean:
	rm -rf obj/ BitEA

# Build by default without cleaning after
all: build
