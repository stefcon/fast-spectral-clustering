# Download and munge the data sets
data: init
	./setup/data.sh
	python3 ./src/preprocess.py

# Set up the environment
.PHONY: init
init:
	./setup/init.sh
	./setup/lib.sh

NVCC = nvcc

# path #
SRC_PATH = src
BUILD_PATH = build
BIN_PATH = $(BUILD_PATH)/bin
BIN_NAME = program

# extensions #
SRC_EXT = cpp
CU_EXT = cu

# code lists #
# Find all source files in the source directory, sorted by
# most recently modified
SOURCES_CPP = $(shell find $(SRC_PATH) -name '*.$(SRC_EXT)' | sort -k 1nr | cut -f2-)
SOURCES_CU = $(shell find $(SRC_PATH) -name '*.$(CU_EXT)' | sort -k 1nr | cut -f2-)
# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
OBJECTS_CPP = $(SOURCES_CPP:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
OBJECTS_CU = $(SOURCES_CU:$(SRC_PATH)/%.$(CU_EXT)=$(BUILD_PATH)/%.o)
OBJECTS = $(OBJECTS_CPP) $(OBJECTS_CU)
# Set the dependency files that will be used to add header dependencies
DEPS = $(OBJECTS:.o=.d)

# flags #
COMPILE_FLAGS = -O3 -std=c++17 -arch=sm_70 -lineinfo -Xcompiler -fopenmp

INCLUDES = -I/usr/include -I/usr/local/cuda/include -I$(HOME)/diplomski/lib -I $(HOME)/diplomski/lib/armadillo/include

LIBS = -L/usr/local/lib -L/usr/local/cuda/lib64 -L$(HOME)/diplomski/lib -L $(HOME)/diplomski/lib/armadillo -llapack -lblas -lcublas -lcudart -lculibos -lcusolver -lgomp
 
INC = $(INCLUDES) $(LIBS)

.PHONY: default_target
default_target: release

.PHONY: release
release: export NVCCFLAGS := $(NVCCFLAGS) $(COMPILE_FLAGS)
release: dirs
	@$(MAKE) all

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJECTS))
	@mkdir -p $(BIN_PATH)

.PHONY: clean
clean:
	@echo "Deleting $(BIN_NAME) symlink"
	@$(RM) $(BIN_NAME)
	@echo "Deleting directories"
	@$(RM) -r $(BUILD_PATH)
	@$(RM) -r $(BIN_PATH)

# checks the executable and symlinks to the output
.PHONY: all
all: $(BIN_PATH)/$(BIN_NAME)
	@echo "Making symlink: $(BIN_NAME) -> $<"
	@$(RM) $(BIN_NAME)
	@ln -s $(BIN_PATH)/$(BIN_NAME) $(BIN_NAME)

# Creation of the executable
$(BIN_PATH)/$(BIN_NAME): $(OBJECTS)
	@echo "Linking: $@"
	$(NVCC) $(OBJECTS) -o $@ ${LIBS}
	

# Add dependency files, if they exist
-include $(DEPS)
 
# Source file rules
# After the first compilation they will be joined with the rules from the
# dependency files to provide header dependencies
$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(NVCC) $(NVCCFLAGS) $(INC) -c $< -o $@

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(CU_EXT)
	@echo "Compiling: $< -> $@"
	$(NVCC) $(NVCCFLAGS) $(INC) -c $< -o $@