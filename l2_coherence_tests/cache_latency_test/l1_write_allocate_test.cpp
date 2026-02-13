#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(cmd)                                                   \
    {                                                                    \
        hipError_t error = cmd;                                          \
        if (error != hipSuccess)                                         \
        {                                                                \
            std::cerr << "Error: '" << hipGetErrorString(error) << "' (" \
                      << error << ") at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                      \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

#define ITERATIONS 256
#define CACHE_LINE_SIZE 128

__global__ void write_allocate_test_kernel(int *data, unsigned long long *latencies, int *loaded_vals)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;

    int value;
    int store_val = 0xDEAD;
    unsigned long long start, end, total;

    // Test 1: Load -> Load (baseline L1 hit)
    int *addr1 = data;
    asm volatile("global_load_dword %0, %1, off\n"
                 "s_waitcnt vmcnt(0)" : "=v"(value) : "v"(addr1) : "memory");

    total = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        start = __builtin_amdgcn_s_memtime();
        asm volatile("global_load_dword %0, %1, off\n"
                     "s_waitcnt vmcnt(0)" : "=v"(value) : "v"(addr1) : "memory");
        end = __builtin_amdgcn_s_memtime();
        total += end - start;
    }
    latencies[0] = total / ITERATIONS;
    loaded_vals[0] = value;

    // Test 2: Store -> Load (is data in L1 after store?)
    int *addr2 = data + CACHE_LINE_SIZE;
    total = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        asm volatile("buffer_inv sc1" ::: "memory");
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");

        asm volatile("global_store_dword %0, %1, off\n"
                     "s_waitcnt vmcnt(0)" : : "v"(addr2), "v"(store_val) : "memory");

        start = __builtin_amdgcn_s_memtime();
        asm volatile("global_load_dword %0, %1, off\n"
                     "s_waitcnt vmcnt(0)" : "=v"(value) : "v"(addr2) : "memory");
        end = __builtin_amdgcn_s_memtime();
        total += end - start;
    }
    latencies[1] = total / ITERATIONS;
    loaded_vals[1] = value;

    // Test 3: Store -> Load -> Load (is data in L1 after load brings it?)
    int *addr3 = data + 2 * CACHE_LINE_SIZE;
    total = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        asm volatile("buffer_inv sc1" ::: "memory");
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");

        asm volatile("global_store_dword %0, %1, off\n"
                     "s_waitcnt vmcnt(0)" : : "v"(addr3), "v"(store_val) : "memory");

        asm volatile("global_load_dword %0, %1, off\n"
                     "s_waitcnt vmcnt(0)" : "=v"(value) : "v"(addr3) : "memory");

        start = __builtin_amdgcn_s_memtime();
        asm volatile("global_load_dword %0, %1, off\n"
                     "s_waitcnt vmcnt(0)" : "=v"(value) : "v"(addr3) : "memory");
        end = __builtin_amdgcn_s_memtime();
        total += end - start;
    }
    latencies[2] = total / ITERATIONS;
    loaded_vals[2] = value;
}

int main()
{
    int *d_data;
    unsigned long long *d_latencies;
    int *d_loaded_vals;

    size_t data_size = 3 * CACHE_LINE_SIZE * sizeof(int);
    HIP_CHECK(hipMalloc(&d_data, data_size));
    HIP_CHECK(hipMemset(d_data, 0, data_size));
    HIP_CHECK(hipMalloc(&d_latencies, 3 * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_loaded_vals, 3 * sizeof(int)));

    hipLaunchKernelGGL(write_allocate_test_kernel,
                       dim3(1), dim3(1), 0, 0,
                       d_data, d_latencies, d_loaded_vals);

    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long h_latencies[3];
    int h_loaded_vals[3];
    HIP_CHECK(hipMemcpy(h_latencies, d_latencies, 3 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_loaded_vals, d_loaded_vals, 3 * sizeof(int), hipMemcpyDeviceToHost));

    std::cout << "=== L1 Write-Allocate Test ===\n\n";
    std::cout << "Latencies (cycles, avg of " << ITERATIONS << " iterations):\n";
    std::cout << "  1. Load -> Load:          " << h_latencies[0] << " (val: 0x" << std::hex << h_loaded_vals[0] << ")\n";
    std::cout << "  2. Store -> Load:         " << std::dec << h_latencies[1] << " (val: 0x" << std::hex << h_loaded_vals[1] << ", expected: 0xDEAD)\n";
    std::cout << "  3. Store -> Load -> Load: " << std::dec << h_latencies[2] << " (val: 0x" << std::hex << h_loaded_vals[2] << ", expected: 0xDEAD)\n";

    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_latencies));
    HIP_CHECK(hipFree(d_loaded_vals));

    return 0;
}
