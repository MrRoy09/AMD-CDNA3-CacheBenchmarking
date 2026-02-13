# Coherence Test 7: WBL2 + Buffer Invalidate + Load

## Test Description

Examines the combined effect of explicit L2 writeback followed by buffer invalidation before performing loads.

## Results
root@7:~# ./wbl2_buffer_inv_load 
Coherent reads: 304 / Incoherent reads: 0 / Wrong reads: 0

## Conclusion
Writeback triggers snoop filters which invalidate local L2 cached line. Buffer Invalidate instruction only operations non-local cached lines.
