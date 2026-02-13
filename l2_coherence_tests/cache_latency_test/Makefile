HIPCC = hipcc
CXXFLAGS = -O1 -g --save-temps
ARCH = --offload-arch=gfx942

all: l1_latency_test l1_write_allocate_test

l1_latency_test: l1_latency_test.cpp
	$(HIPCC) $(CXXFLAGS) $(ARCH) -o $@ $<
	rm -f *.bc *.hipi *.s *.hipfb *.out *.out.resolution.txt *-host-*.o

l1_write_allocate_test: l1_write_allocate_test.cpp
	$(HIPCC) $(CXXFLAGS) $(ARCH) -o $@ $<
	rm -f *.bc *.hipi *.s *.hipfb *.out *.out.resolution.txt *-host-*.o

clean:
	rm -f l1_latency_test l1_write_allocate_test
	rm -f *.bc *.hipi *.s *.hipfb *.out *.out.resolution.txt *.o

.PHONY: all clean