# Makefile for Coherence Test 3: Scoped Store + NT Load

HIPCC = hipcc
CXXFLAGS = -O0 -g --save-temps
ARCH = --offload-arch=gfx942
SRCDIR = .
TARGET = scoped_store_nt_load
SRC = $(SRCDIR)/scoped_store_nt_load.cpp
GPU_OBJ = scoped_store_nt_load-hip-amdgcn-amd-amdhsa-gfx942.o

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