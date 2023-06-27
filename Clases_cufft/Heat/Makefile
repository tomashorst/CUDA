# Binary file
BIN = heatcufft

# Flags
CUDA_INSTALL_PATH       = /usr/local/cuda
CUDA_LIB_PATH           = $(CUDA_INSTALL_PATH)/lib64
CUDASDK_LIB_PATH        = $(CUDASDK_INSTALL_PATH)/C/lib64

INCLUDES += -I. -I$(CUDA_INSTALL_PATH)/include 
LIBS                    = -L$(CUDA_LIB_PATH)
CFLAGS                  = #-O3
LDFLAGS                 = -lrt -lm -lcudart -lcufft
NVCC_FLAGS              = -O3 --ptxas-options=-v --use_fast_math -ftz=true 

# Default Values
LX			= 256
LY			= 256
T_RUN			= 50000
T_DATA			= 100
C_BETA			= 1.f

# Simulation Parameters
PARAMETERS              = -DLX=$(LX) -DLY=$(LY) -DT_RUN=$(T_RUN) -DT_DATA=$(T_DATA) -DC_BETA=$(C_BETA) 

# Compilers
NVCC                    = nvcc
GCC                     = g++
LINKER                  = g++

# Files
C_SOURCES               =
CU_SOURCES              = main.cu
HEADERS                 = heat.cuh ppmimages.cuh
C_OBJS                  = $(patsubst %.c, %.o, $(C_SOURCES))
CU_OBJS                 = $(patsubst %.cu, %.o, $(CU_SOURCES))


# Rules
$(BIN): clean $(C_OBJS) $(CU_OBJS) $(HEADERS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(C_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

$(C_OBJS): $(C_SOURCES) $(HEADERS)
	$(CC) -c $(C_SOURCES) $(CFLAGS) $(INCLUDES)

$(CU_OBJS): $(CU_SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $(CU_SOURCES) $(PARAMETERS) $(INCLUDES)

run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ./$(BIN) > $(OFILE) & 

clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log
                                                         
