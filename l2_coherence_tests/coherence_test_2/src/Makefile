# Makefile for Coherence Test 2: Load with NT bit

HIPCC = hipcc
CXXFLAGS = -O1 -g --save-temps
ARCH = --offload-arch=gfx942
SRCDIR = .

TARGETS = load_nt load_nt_fixed

all: $(TARGETS)

load_nt: $(SRCDIR)/load_nt.cpp
	$(HIPCC) $(CXXFLAGS) $(ARCH) $< -o $@
	rm -f *.bc *.hipi *.s *.hipfb *.out *.out.resolution.txt *-host-*.o

load_nt_fixed: $(SRCDIR)/load_nt_fixed.cpp
	$(HIPCC) $(CXXFLAGS) $(ARCH) $< -o $@
	rm -f *.bc *.hipi *.s *.hipfb *.out *.out.resolution.txt *-host-*.o

clean:
	rm -f $(TARGETS)
	rm -f *.bc *.hipi *.s *.hipfb *.out *.out.resolution.txt *.o

run: load_nt
	./load_nt

run_fixed: load_nt_fixed
	./load_nt_fixed

.PHONY: all clean run run_fixed