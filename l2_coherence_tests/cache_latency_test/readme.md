# Cache Latency and Write-Allocate Tests

## Test Description

This directory contains two tests to characterize the L1 cache behavior on CDNA3:

### 1. L1 Latency Test (`l1_latency_test`)
Measures the latency of different types of loads to determine which cache levels they access:
- **1st load (memory, cold):** Measures latency of a cold load (likely DRAM).
- **2nd load (L1 hit):** Measures latency of subsequent loads to the same address, which should hit in L1.
- **3rd load (NT):** Measures latency of a load with the `nt` (non-temporal) bit set, which should bypass L1 and hit in L2.
- **4th load (sc1):** Measures latency of a load with the `sc1` (system scope) bit set.
- **5th load (after inv sc1):** Measures latency of a load after a `buffer_inv sc1` instruction.

### 2. L1 Write-Allocate Test (`l1_write_allocate_test`)
Determines if the L1 cache is write-allocate or write-no-allocate by comparing latencies:
- **Test 1 (Load -> Load):** Baseline L1 hit latency.
- **Test 2 (Store -> Load):** Measures latency of a load immediately after a store to the same address.
- **Test 3 (Store -> Load -> Load):** Measures latency of a second load after a store and an initial load.

## Results

```
root@7:~# ./l1_latency_test 
Load latencies (cycles, avg of 256 iterations):
  1st load (memory, cold):    1212
  2nd load (L1 hit):          112
  3rd load (NT):              224
  4th load (sc1):             224
  5th load (after inv sc1):   212

root@7:~# ./l1_write_allocate_test 
=== L1 Write-Allocate Test ===

Latencies (cycles, avg of 256 iterations):
  1. Load -> Load:          127 (val: 0x0)
  2. Store -> Load:         208 (val: 0xdead, expected: 0xDEAD)
  3. Store -> Load -> Load: 116 (val: 0xdead, expected: 0xDEAD)
```

## Conclusion

- **L1 Hit Latency:** ~112 cycles.
- **L2 Hit Latency (NT load):** ~224 cycles.
- **Non-Temporal (NT) and Scoped (sc1) loads** bypass the L1 cache and hit in the L2 cache.
- **L1 Cache Policy:** The `Store -> Load` latency (~208 cycles) matches the L2 latency, while `Store -> Load -> Load` (~116 cycles) matches the L1 latency. **L1 cache is write-no-allocate**.