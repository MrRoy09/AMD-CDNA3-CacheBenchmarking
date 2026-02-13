# Coherence Test 2: Load with NT bit (Bypass L1)

## Test Description

Examines cache coherence behavior by loading from L2 (uses NT bit to bypass L1). The test is in two parts:

- XCCs keep polling until they see the updated value or until max iterations are completed. Reports the number of iterations it took to observe the updated value.
- XCCs iterate and load for a fixed number of iterations and report the value they observed.

## Results
```
root@7:~# ./load_nt

Iterations needed to see value 42 (by XCC):

XCC 0: 38 blocks
  Min iterations: 8
  Max iterations: 9
  Avg iterations: 8

XCC 1: 38 blocks
  Min iterations: 12
  Max iterations: 14
  Avg iterations: 13

XCC 2: 38 blocks
  Min iterations: 13
  Max iterations: 15
  Avg iterations: 14

XCC 3: 38 blocks
  Min iterations: 10
  Max iterations: 12
  Avg iterations: 11

XCC 4: 38 blocks
  Min iterations: 7
  Max iterations: 8
  Avg iterations: 7

XCC 5: 38 blocks
  Min iterations: 6
  Max iterations: 7
  Avg iterations: 6

XCC 6: 38 blocks
  Min iterations: 1
  Max iterations: 2
  Avg iterations: 1

XCC 7: 38 blocks
  Min iterations: 12
  Max iterations: 13
  Avg iterations: 12
```

```
root@7:~# ./load_nt_fixed 
Fixed 200000 iterations NT load test (by XCC):
Checking if the final value read was 42

XCC 0: 0/38 blocks saw 42
XCC 1: 1/38 blocks saw 42
XCC 2: 1/38 blocks saw 42
XCC 3: 0/38 blocks saw 42
XCC 4: 1/38 blocks saw 42
XCC 5: 1/38 blocks saw 42
XCC 6: 38/38 blocks saw 42
XCC 7: 1/38 blocks saw 42
```

## Conclusion

In test 1, the workgroup making the store was scheduled on XCC 6. All the CUs on XCC 6 observed the updated value almost immediately and completed execution. Once all scheduled workgroups finished execution, the dirty line was written back, which caused the snoop filters to trigger invalidation of L2 cachelines in other XCCs' L2 caches. 

All other CUs saw the updated value soon after.

In test 2, the XCC with the updated value was not able to finish execution due to the fixed iterations. Hence, most XCCs observed the stale value. The few blocks on other XCCs seeing updated values are due to the fact that the XCC completed execution and wrote back dirty lines while a few workgroups were still busy.