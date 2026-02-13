# Coherence Test 3: Device-Scoped Store + NT Load

## Test Description

Examines cache coherence behavior with device-scoped store (SC1) and non-temporal loads (NT bit) that bypass L1 cache.

## Results
```
root@7:~# ./scoped_store_nt_load 
Coherent reads: 304 / Incoherent reads: 0 / Wrong reads : 0
```

## Conclusion
Device-scoped stores also trigger the snoop filter, which invalidates the L2 cache lines in other XCDs.
