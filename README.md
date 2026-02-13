# CDNA3 L2 Coherence Benchmarks

This repository contains benchmarks and tests for characterizing the L2 cache coherence behavior on AMD CDNA3 architecture (MI300X).

## Contents

### Cache Characterization
- **cache_latency_test**: Measures L1 and L2 hit latencies and determines L1 write-allocation policy.

### Coherence Tests
- **coherence_test_1**: Regular load/store behavior.
- **coherence_test_2**: Load with NT bit (L1 bypass) and its interaction with snoop filters.
- **coherence_test_3**: Device-scoped store (SC1) + NT load interaction.
- **coherence_test_4**: Regular store + explicit L2 writeback (WBL2) + NT load.
- **coherence_test_5**: Regular store + device-scoped load (SC1).
- **coherence_test_6**: Behavior of `buffer_inv sc1` on cache lines.
- **coherence_test_7**: Interaction between WBL2 and buffer invalidation.
