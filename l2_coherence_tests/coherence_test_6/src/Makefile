# Makefile for Coherence Test 6: Buffer Invalidate SC1 Test

HIPCC = hipcc
CXXFLAGS = -O0 -g --save-temps
ARCH = --offload-arch=gfx942
SRCDIR = .
TARGET = buffer_inv_test
SRC = $(SRCDIR)/buffer_inv_test.cpp
GPU_OBJ = buffer_inv_test-hip-amdgcn-amd-amdhsa-gfx942.o

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
