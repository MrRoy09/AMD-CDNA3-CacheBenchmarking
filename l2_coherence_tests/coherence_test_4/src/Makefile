# Makefile for Coherence Test 4: Regular Store + WBL2 + NT Load

HIPCC = hipcc
CXXFLAGS = -O0 -g --save-temps
ARCH = --offload-arch=gfx942
SRCDIR = .
TARGET = regular_store_wbl2_nt_load
SRC = $(SRCDIR)/regular_store_wbl2_nt_load.cpp
GPU_OBJ = regular_store_wbl2_nt_load-hip-amdgcn-amd-amdhsa-gfx942.o

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) $(ARCH) $(SRC) -o $(TARGET)
	rm -f *.bc *.hipi *.s *.hipfb *.out *.out.resolution.txt *-host-*.o

clean:
	rm -f $(TARGET) $(GPU_OBJ)
	rm -f *.bc *.hipi *.s *.hipfb *.out *.out.resolution.txt *.o

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
