# Makefile for Coherence Test 5: Regular Store + Scoped Load (SC1)

HIPCC = hipcc
CXXFLAGS = -O0 -g --save-temps
ARCH = --offload-arch=gfx942
SRCDIR = .
TARGET = regular_store_scoped_load
SRC = $(SRCDIR)/regular_store_scoped_load.cpp
GPU_OBJ = regular_store_scoped_load-hip-amdgcn-amd-amdhsa-gfx942.o

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
