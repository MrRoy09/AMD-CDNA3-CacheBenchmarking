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

__global__ void latency_test_kernel(int *data, unsigned long long *latencies)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid != 0) return;  // Only one thread measures

    int *ptr = data;
    int value;
    unsigned long long start, end, total;

    // First load - should come from memory (cold)
    start = __builtin_amdgcn_s_memtime();
    asm volatile("global_load_dword %0, %1, off\n"
                 "s_waitcnt vmcnt(0)" : "=v"(value) : "v"(ptr) : "memory");
    end = __builtin_amdgcn_s_memtime();
    latencies[0] = end - start;

    // Second load - L1 hit (averaged)
    total = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        start = __builtin_amdgcn_s_memtime();
        asm volatile("global_load_dword %0, %1, off\n"
                     "s_waitcnt vmcnt(0)" : "=v"(value) : "v"(ptr) : "memory");
        end = __builtin_amdgcn_s_memtime();
        total += end - start;
    }
    latencies[1] = total / ITERATIONS;

    // Third load - NT load (averaged)
    total = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        start = __builtin_amdgcn_s_memtime();
        asm volatile("global_load_dword %0, %1, off nt\n"
                     "s_waitcnt vmcnt(0)" : "=v"(value) : "v"(ptr) : "memory");
        end = __builtin_amdgcn_s_memtime();
        total += end - start;
    }
    latencies[2] = total / ITERATIONS;

    // Fourth load - sc1 scope (averaged)
    total = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        start = __builtin_amdgcn_s_memtime();
        asm volatile("global_load_dword %0, %1, off sc1\n"
                     "s_waitcnt vmcnt(0)" : "=v"(value) : "v"(ptr) : "memory");
        end = __builtin_amdgcn_s_memtime();
        total += end - start;
    }
    latencies[3] = total / ITERATIONS;

    // Fifth load - after buffer_inv sc1 but not including time for invalidation to complete (averaged)
    total = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        asm volatile("buffer_inv sc1" ::: "memory");
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        start = __builtin_amdgcn_s_memtime();
        asm volatile("global_load_dword %0, %1, off\n"
                     "s_waitcnt vmcnt(0)" : "=v"(value) : "v"(ptr) : "memory");
        end = __builtin_amdgcn_s_memtime();
        total += end - start;
    }
    latencies[4] = total / ITERATIONS;

    // Prevent compiler from optimizing away the loads
    if (value == -999999) latencies[0] = 0;
}

int main()
{
    int *d_data;
    unsigned long long *d_latencies;

    HIP_CHECK(hipMalloc(&d_data, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_latencies, 5 * sizeof(unsigned long long)));

    int init_value = 42;
    HIP_CHECK(hipMemcpy(d_data, &init_value, sizeof(int), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(latency_test_kernel,
                       dim3(1),
                       dim3(1),
                       0, 0,
                       d_data,
                       d_latencies);

    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long h_latencies[5];
    HIP_CHECK(hipMemcpy(h_latencies, d_latencies, 5 * sizeof(unsigned long long), hipMemcpyDeviceToHost));

    std::cout << "Load latencies (cycles, avg of " << ITERATIONS << " iterations):\n";
    std::cout << "  1st load (memory, cold):    " << h_latencies[0] << "\n";
    std::cout << "  2nd load (L1 hit):          " << h_latencies[1] << "\n";
    std::cout << "  3rd load (NT):              " << h_latencies[2] << "\n";
    std::cout << "  4th load (sc1):             " << h_latencies[3] << "\n";
    std::cout << "  5th load (after inv sc1):   " << h_latencies[4] << "\n";

    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_latencies));

    return 0;
}