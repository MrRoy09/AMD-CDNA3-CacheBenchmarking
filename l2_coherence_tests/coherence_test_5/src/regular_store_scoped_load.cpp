#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>

#define HIP_CHECK(cmd)                                                   \
    {\
        hipError_t error = cmd;                                          \
        if (error != hipSuccess)                                         \
        {\
            std::cerr << "Error: '" << hipGetErrorString(error) << "' (" \
                      << error << ") at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                      \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

// XCC_ID register (register 20), XCC_ID is bits 3:0
#define XCC_ID 20
#define XCC_ID_SIZE 4
#define XCC_ID_OFFSET 0
#define GETREG_IMMED(SIZE, OFFSET, HWREG) (((SIZE) << 11) | ((OFFSET) << 6) | (HWREG))

#define READ_ITERATIONS 4096

__global__ void regular_store_scoped_load_kernel(int *shared_value, int *results, int *xcc_ids)
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

    // Block 0, thread 0 performs a REGULAR store (no scope, no cache ops)
    if (block_id == 0 && tid == 0)
    {
        shared_value[0] = 42;
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    __syncthreads();

    // All blocks continuously poll with sc1 loads
    if (tid == 0)
    {
        int value = 0;
        for (int i = 0; i < READ_ITERATIONS; ++i) {
            asm volatile(
                "global_load_dword %0, %1, off sc1\n"
                "s_waitcnt vmcnt(0)\n"
                : "=v"(value)
                : "v"(shared_value)
                : "memory");
        }

        // Store the final value read
        results[block_id] = value;
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
    int *d_results;
    int *d_xcc_ids;

    HIP_CHECK(hipMalloc(&d_shared_value, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_results, num_blocks * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_xcc_ids, num_blocks * sizeof(int)));

    int old_value = 99;
    HIP_CHECK(hipMemcpy(d_shared_value, &old_value, sizeof(int), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(regular_store_scoped_load_kernel,
                       dim3(num_blocks),
                       dim3(threads_per_block),
                       0, 0,
                       d_shared_value,
                       d_results,
                       d_xcc_ids);

    HIP_CHECK(hipDeviceSynchronize());

    int *h_results = new int[num_blocks];
    int *h_xcc_ids = new int[num_blocks];
    
    HIP_CHECK(hipMemcpy(h_results, d_results, num_blocks * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_xcc_ids, d_xcc_ids, num_blocks * sizeof(int), hipMemcpyDeviceToHost));

    // Group results by XCC
    std::cout << "Scoped Load (sc1) Test Results (" << READ_ITERATIONS << " iterations):\n";
    std::cout << "Final value analysis (Coherent=42, Incoherent=99)\n\n";

    for (int xcc = 0; xcc < 8; xcc++)
    {
        int block_count = 0;
        int coherent_count = 0;
        int incoherent_count = 0;
        int other_count = 0;

        for (int i = 0; i < num_blocks; i++)
        {
            if (h_xcc_ids[i] == xcc)
            {
                block_count++;
                if (h_results[i] == 42) {
                    coherent_count++;
                } else if (h_results[i] == 99) {
                    incoherent_count++;
                } else {
                    other_count++;
                }
            }
        }

        if (block_count > 0)
        {
            std::cout << "XCC " << xcc << ": " << block_count << " blocks\n";
            std::cout << "  Coherent (42): " << coherent_count << "\n";
            std::cout << "  Incoherent (99): " << incoherent_count << "\n";
            if (other_count > 0) {
                std::cout << "  Other values: " << other_count << "\n";
            }
            std::cout << "\n";
        }
    }

    delete[] h_results;
    delete[] h_xcc_ids;
    HIP_CHECK(hipFree(d_shared_value));
    HIP_CHECK(hipFree(d_results));
    HIP_CHECK(hipFree(d_xcc_ids));

    return 0;
}
