# Coherence Test 4: Regular Store + WBL2 + NT Load

## Test Description

Examines cache coherence behavior when using explicit L2 writeback instruction after a regular store, combined with non-temporal loads (L1 bypass).


## Results
```
root@7:~# ./regular_store_wbl2_nt_load 
Coherent reads: 304 / Incoherent reads: 0 / Wrong reads : 0
```

## Conclusion

Writeback triggers the snoop filters, which invalidate L2 lines of other XCDs.
