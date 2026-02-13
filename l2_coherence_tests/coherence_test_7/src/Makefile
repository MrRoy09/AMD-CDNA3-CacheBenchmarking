# Makefile for Coherence Test 7: WBL2 + Buffer Inv + Load

HIPCC = hipcc
CXXFLAGS = -O0 -g --save-temps
ARCH = --offload-arch=gfx942
SRCDIR = .
TARGET = wbl2_buffer_inv_load
SRC = $(SRCDIR)/wbl2_buffer_inv_load.cpp
GPU_OBJ = wbl2_buffer_inv_load-hip-amdgcn-amd-amdhsa-gfx942.o

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
