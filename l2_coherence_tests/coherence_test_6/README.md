# Coherence Test 6: Buffer Invalidate SC1 Behavior Test

## Test Description

This test examines whether `buffer_inv sc1` invalidates L2 cache lines across all XCDs, particularly testing if it affects hardware-coherency-controlled cache lines.

## Results
```
root@7:~# ./buffer_inv_test 
GPU: AMD Instinct MI300X VF
Total CUs: 304
42 read by: 38 / 99 read by: 266 / Wrong reads: 0
```

## Conclusion
> The buffer invalidate did not invalidate the L2 cacheline containing the updated value, nor did it writeback the dirty L2 line containing the updated value.

> Hence, buffer invalidate is only applicable for non-local memory cachelines.
