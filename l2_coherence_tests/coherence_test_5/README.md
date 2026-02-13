# Coherence Test 5: Regular Store + Device-Scoped Load (SC1)

## Test Description

Examines cache coherence behavior with regular store and device-scoped loads that bypass L1 and L2 caches.

## Results
```
root@7:~# ./regular_store_scoped_load 
Scoped Load (sc1) Test Results (4096 iterations):
Final value analysis (Coherent=42, Incoherent=99)

XCC 0: 38 blocks
  Coherent (42): 1
  Incoherent (99): 37

XCC 1: 38 blocks
  Coherent (42): 0
  Incoherent (99): 38

XCC 2: 38 blocks
  Coherent (42): 1
  Incoherent (99): 37

XCC 3: 38 blocks
  Coherent (42): 2
  Incoherent (99): 36

XCC 4: 38 blocks
  Coherent (42): 1
  Incoherent (99): 37

XCC 5: 38 blocks
  Coherent (42): 1
  Incoherent (99): 37

XCC 6: 38 blocks
  Coherent (42): 38
  Incoherent (99): 0

XCC 7: 38 blocks
  Coherent (42): 2
  Incoherent (99): 36
```

## Conclusion

Regular stores are not written back even when a load with device scope is requested. This is why most of the scoped loads are still fetching stale values. The few that are fetching the updated value are because XCC 6 has finished execution and written back dirty L2 lines.
