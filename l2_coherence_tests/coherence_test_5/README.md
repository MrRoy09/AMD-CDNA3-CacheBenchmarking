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

Dirty L2 cachelines are not written back even when a load with device scope is requested. The few workgroups in other XCCs that are fetching the updated value are because XCC 6 has finished execution and written back dirty L2 lines (Write-back is triggered by completion of all scheduled workgroups)
