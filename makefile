ARCH ?= sm_60
HOST_COMP ?= mpicxx

NVCC := nvcc
NVCC_FLAGS := -O3 -std=c++14 -arch=$(ARCH) -Xcompiler -O3

SRC := main.cpp kernels.cu
TARGET := main.o

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -ccbin $(HOST_COMP) -o $@ $(SRC)

clean:
	rm -f $(TARGET)

