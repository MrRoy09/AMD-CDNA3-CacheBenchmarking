#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>

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

#define XCC_ID 20
#define XCC_ID_SIZE 4
#define XCC_ID_OFFSET 0
#define GETREG_IMMED(SIZE, OFFSET, HWREG) (((SIZE) << 11) | ((OFFSET) << 6) | (HWREG))

__global__ void regular_load_store_kernel(int *shared_value, unsigned long long *iterations, int *xcc_ids)
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    // All workgroups read the old value to pull into L2
    int old_value;
    for (int i = 0; i < 256; i++)
    {
        asm volatile("global_load_dword %0, %1, off" : "=v"(old_value) : "v"(shared_value) : "memory");
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    __syncthreads();

    // Block 0, thread 0 performs a REGULAR store with default scope
    if (block_id == 0 && tid == 0)
    {
        shared_value[0] = 42;
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    __syncthreads();

    // All blocks continuously poll with REGULAR loads until they see 42
    if (tid == 0)
    {
        int value;
        unsigned long long count = 0;
        const unsigned long long max_iterations = 100000000ULL;  // 100M max

        do {
            asm volatile(
                "global_load_dword %0, %1, off\n"
                "s_waitcnt vmcnt(0)\n"
                : "=v"(value)
                : "v"(shared_value)
                : "memory");
            count++;
        } while (value != 42 && count < max_iterations);

        iterations[block_id] = count;
        xcc_ids[block_id] = __builtin_amdgcn_s_getreg(GETREG_IMMED(XCC_ID_SIZE - 1, XCC_ID_OFFSET, XCC_ID));
    }
}

int main(int argc, char **argv)
{
    int device = 0;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    int threads_per_block = 1024;
    int num_blocks = props.multiProcessorCount;

    int *d_shared_value;
    unsigned long long *d_iterations;
    int *d_xcc_ids;

    HIP_CHECK(hipMalloc(&d_shared_value, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_iterations, num_blocks * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_xcc_ids, num_blocks * sizeof(int)));

    int old_value = 99;
    HIP_CHECK(hipMemcpy(d_shared_value, &old_value, sizeof(int), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(regular_load_store_kernel,
                       dim3(num_blocks),
                       dim3(threads_per_block),
                       0, 0,
                       d_shared_value,
                       d_iterations,
                       d_xcc_ids);

    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long *h_iterations = new unsigned long long[num_blocks];
    int *h_xcc_ids = new int[num_blocks];
    HIP_CHECK(hipMemcpy(h_iterations, d_iterations, num_blocks * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_xcc_ids, d_xcc_ids, num_blocks * sizeof(int), hipMemcpyDeviceToHost));

    std::cout << "Iterations needed to see value 42 (by XCC) - using regular load:\n\n";

    for (int xcc = 0; xcc < 8; xcc++)
    {
        unsigned long long min_iter = ULLONG_MAX, max_iter = 0, total_iter = 0;
        int count = 0;
        int timed_out = 0;

        for (int i = 0; i < num_blocks; i++)
        {
            if (h_xcc_ids[i] == xcc)
            {
                count++;
                if (h_iterations[i] >= 100000000ULL)
                    timed_out++;
                else
                {
                    if (h_iterations[i] < min_iter) min_iter = h_iterations[i];
                    if (h_iterations[i] > max_iter) max_iter = h_iterations[i];
                    total_iter += h_iterations[i];
                }
            }
        }

        if (count > 0)
        {
            std::cout << "XCC " << xcc << ": " << count << " blocks\n";
            if (timed_out > 0)
                std::cout << "  Timed out (never saw 42): " << timed_out << " blocks\n";
            if (count - timed_out > 0)
            {
                std::cout << "  Min iterations: " << min_iter << "\n";
                std::cout << "  Max iterations: " << max_iter << "\n";
                std::cout << "  Avg iterations: " << (total_iter / (count - timed_out)) << "\n";
            }
            std::cout << "\n";
        }
    }

    delete[] h_iterations;
    delete[] h_xcc_ids;
    HIP_CHECK(hipFree(d_shared_value));
    HIP_CHECK(hipFree(d_iterations));
    HIP_CHECK(hipFree(d_xcc_ids));

    return 0;
}
