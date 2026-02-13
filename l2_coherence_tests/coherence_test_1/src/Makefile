# Makefile for Coherence Test 1: Regular Load/Store

HIPCC = hipcc
CXXFLAGS = -O1 -g --save-temps
ARCH = --offload-arch=gfx942
SRCDIR = .
TARGET = regular_load_store
SRC = $(SRCDIR)/regular_load_store.cpp
GPU_OBJ = regular_load_store-hip-amdgcn-amd-amdhsa-gfx942.o

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