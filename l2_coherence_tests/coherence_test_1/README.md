# Coherence Test 1: Regular Load/Store

Examines cache coherency behaviour for regular load/store operations (no scope, no cache bypass)

## Results
```
root@7:~# ./regular_load_store 
Iterations needed to see value 42 (by XCC) - using regular load:

XCC 0: 38 blocks
  Timed out (never saw 42): 38 blocks

XCC 1: 38 blocks
  Timed out (never saw 42): 38 blocks

XCC 2: 38 blocks
  Timed out (never saw 42): 38 blocks

XCC 3: 38 blocks
  Timed out (never saw 42): 38 blocks

XCC 4: 38 blocks
  Timed out (never saw 42): 38 blocks

XCC 5: 38 blocks
  Timed out (never saw 42): 38 blocks

XCC 6: 38 blocks
  Timed out (never saw 42): 37 blocks
  Min iterations: 1
  Max iterations: 1
  Avg iterations: 1

XCC 7: 38 blocks
  Timed out (never saw 42): 38 blocks
```

## Conclusion

L1 is not coherent.