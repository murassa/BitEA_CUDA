# Paths and Flags
CUDA_PATH ?= /usr/local/cuda
SEARCH_PATH = include

# Compiler Flags
CFLAGS = -Wall -std=gnu17 -I$(SEARCH_PATH) -lstdc++
LINKFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lm
INCLUDES = -I$(CUDA_PATH)/include -I$(SEARCH_PATH)
CUDAFLAGS = -I$(SEARCH_PATH) -std=c++17
NVCC_ARCH = -arch=sm_50  # Change 'sm_50' to your desired architecture

# Debug Flags
ifdef DEBUG
  CFLAGS += -g -D WITH_DEBUG=1
  NVCCFLAGS += -G -D WITH_DEBUG=1
else
  CFLAGS += -O3
  NVCCFLAGS += -O3
endif

# Source Files and Object Files
SOURCES := $(wildcard src/*.c)
CUDA_SOURCES := $(wildcard src/*.cu)
OBJECTS := $(SOURCES:src/%.c=obj/c/%.o)
CUDA_OBJECTS := $(CUDA_SOURCES:src/%.cu=obj/cu/%.o)

# Check Environment
check-env:
	@nvcc --version || (echo "CUDA is not installed or not in PATH" && exit 1)
	@gcc --version || (echo "GCC is not installed or not in PATH" && exit 1)

# Compile C files with gcc
obj/c/%.o: src/%.c
	@mkdir -p $(@D)
	gcc $(CFLAGS) -c $< -o $@ $(INCLUDES)

# Compile CUDA files with nvcc
obj/cu/%.o: src/%.cu
	@mkdir -p $(@D)
	nvcc $(NVCC_ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDES)

# Link all object files using nvcc
build: check-env $(OBJECTS) $(CUDA_OBJECTS)
	nvcc $(NVCC_ARCH) $(NVCCFLAGS) $(OBJECTS) $(CUDA_OBJECTS) -o BitEA $(LINKFLAGS)

# Clean up object files and executable
clean:
	rm -rf obj/ BitEA

# Build by default
all: build
