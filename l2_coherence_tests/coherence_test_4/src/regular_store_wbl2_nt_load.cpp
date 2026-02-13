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

__global__ void regular_store_wbl2_nt_load_kernel(int *shared_value, int *results)
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    // All workgroups read the old value to pull into L2
    // Use inline asm to force global_load_dword (not scalar load)
    int old_value;
    for (int i = 0; i < 256; i++)
    {
        asm volatile("global_load_dword %0, %1, off" : "=v"(old_value) : "v"(shared_value) : "memory");
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    __syncthreads();

    // Block 0, thread 0 performs regular store + buffer_wbl2
    if (block_id == 0 && tid == 0)
    {
        shared_value[0] = 42;
        asm volatile("buffer_wbl2 sc1" ::: "memory");
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    __syncthreads();

    // All threads perform load with NT bit 256 times (bypass L1)
    int new_value;
    for (int i = 0; i < 256; i++)
    {
        asm volatile(
            "global_load_dword %0, %1, off nt\n"
            "s_waitcnt vmcnt(0)\n"
            : "=v"(new_value)
            : "v"(shared_value)
            : "memory");
    }

    // Store what each workgroup observed
    if (tid == 0)
    {
        results[block_id] = new_value;
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
    int *d_results;

    HIP_CHECK(hipMalloc(&d_shared_value, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_results, num_blocks * sizeof(int)));

    int old_value = 99;
    HIP_CHECK(hipMemcpy(d_shared_value, &old_value, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_results, 0, num_blocks * sizeof(int)));

    hipLaunchKernelGGL(regular_store_wbl2_nt_load_kernel,
                       dim3(num_blocks),
                       dim3(threads_per_block),
                       0, 0,
                       d_shared_value,
                       d_results);

    HIP_CHECK(hipDeviceSynchronize());

    int *h_results = new int[num_blocks];
    HIP_CHECK(hipMemcpy(h_results, d_results, num_blocks * sizeof(int), hipMemcpyDeviceToHost));

    int coherent = 0;
    int incoherent = 0;
    int wrong = 0;

    for (int i = 0; i < num_blocks; i++)
    {
        if (h_results[i] == 42)
        {
            coherent++;
        }
        else if (h_results[i] == 99)
        {
            incoherent++;
        }
        else
        {
            wrong++;
        }
    }

    std::cout << "Coherent reads: " << coherent << " / Incoherent reads: " << incoherent << " / Wrong reads : " << wrong << std::endl;

    delete[] h_results;
    HIP_CHECK(hipFree(d_shared_value));
    HIP_CHECK(hipFree(d_results));

    return 0;
}