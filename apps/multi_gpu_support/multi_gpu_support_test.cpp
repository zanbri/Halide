#include <stdio.h>

#ifdef _WIN32
// This test requires weak linkage
int main(int argc, char **argv) {
  printf("Skipping test on windows\n");
  return 0;
}
#else

#include "HalideRuntime.h"
#include "HalideBuffer.h"

#include "multi_gpu_support.h"

using namespace Halide::Runtime;

/* The purpose of the script is to test whether 2 GPUs can run 2 instances of a kernel
 * simultaneously. */

const int W = 1920, H = 1080;

#if defined(TEST_CUDA)
// Implement CUDA custom contexts.

#include <cuda.h>

CUcontext cuda_ctx1 = nullptr;
CUcontext cuda_ctx2 = nullptr;

int init_context() {
    
    // Initialize CUDA
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        printf("cuInit failed (%d)\n", err);
        return err;
    }
    
    // Make sure we have a device
    int deviceCount = 0;
    err = cuDeviceGetCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);
    
    if (err != CUDA_SUCCESS) {
        printf("cuGetDeviceCount failed (%d)\n", err);
        return err;
    }
    if (deviceCount < 2) {
        printf("This program requires at least 2 CUDA devices.\n");
        return CUDA_ERROR_NO_DEVICE;
    }
    
    CUdevice dev1;
    CUdevice dev2;
    CUresult status;
    
    // Get devices
    status = cuDeviceGet(&dev1, 0);
    if (status != CUDA_SUCCESS) {
        printf("Unable to get CUDA device 1.\n");
        return status;
    }
    status = cuDeviceGet(&dev2, 1);
    if (status != CUDA_SUCCESS) {
        printf("Unable to get CUDA device 2.\n");
        return status;
    }
    
    // Create contexts
    err = cuCtxCreate(&cuda_ctx1, 0, dev1);
    if (err != CUDA_SUCCESS) {
        printf("cuCtxCreate failed (%d)\n", err);
        return err;
    }
    err = cuCtxCreate(&cuda_ctx2, 0, dev2);
    if (err != CUDA_SUCCESS) {
        printf("cuCtxCreate failed (%d)\n", err);
        return err;
    }
    
    printf("Created CUDA context 1: %p\n", cuda_ctx1);
    printf("Created CUDA context 2: %p\n", cuda_ctx2);
    
    return 0;
}

void destroy_context() {
    printf("Destroying CUDA context 1: %p\n", cuda_ctx1);
    printf("Destroying CUDA context 2: %p\n", cuda_ctx2);
    cuCtxDestroy(cuda_ctx1);
    cuCtxDestroy(cuda_ctx2);
    cuda_ctx1 = nullptr;
    cuda_ctx2 = nullptr;
}

// These functions replace the acquire/release implementation in src/runtime/cuda.cpp.
// Since we don't parallelize access to the GPU in the schedule, we don't need synchronization
// in our implementation of these functions.
extern "C" int halide_cuda_acquire_context(void *user_context, CUcontext *ctx, bool create=true) {
    
    if (user_context) {
        int* int_user_context = static_cast<int*>(user_context);
        int selected_gpu = *int_user_context;
        printf("selected_gpu: %d\n", selected_gpu);
        if (selected_gpu == 1) {
            *ctx = cuda_ctx1;
        } else {
            *ctx = cuda_ctx2;
        }
    } else {
        printf("user_context not found.\n");
        *ctx = cuda_ctx1;
    }
            
    return 0;
}

extern "C" int halide_cuda_release_context(void *user_context) {
    return 0;
}

#endif

int main(int argc, char **argv) {
    
    // Define number of times the kernels will be run. This is useful to run many iterations,
    // during which time one can check nvidia-smi, and ensure both devices are being utilized.
    int n_times = 1000;
    if (argc > 1) { n_times = atoi(argv[1]); }
    printf("n_times: %d\n", n_times);
    
    // Initialize the runtime specific GPU contexts
    int ret = init_context();
    if (ret != 0) { return ret; }
    
    // Define input buffer
    Buffer<float> input(W, H);
    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = (float)(x * y);
        }
    }
    
    Buffer<float> output1(W, H);
    Buffer<float> output2(W, H);
    
    const int device1 = 1;
    const int device2 = 2;
    const void* user_context1 = &device1;
    const void* user_context2 = &device2;
    
    for (int i=0; i<n_times; ++i) {
        
        // Run on device 1
        input.set_host_dirty();
        multi_gpu_support(user_context1, input, output1);
        output1.copy_to_host();
        
        // Run on device 2
        input.set_host_dirty();
        multi_gpu_support(user_context2, input, output2);
        output2.copy_to_host();
        
    }
    
    // Verify results from both GPUs
    for (int y = 0; y < output1.height(); y++) {
        for (int x = 0; x < output1.width(); x++) {
            // Compare with device 1 output
            if (input(x, y) * 2.0f + 1.0f != output1(x, y)) {
                printf("Error at (%d, %d): %f != %f\n", x, y, input(x, y) * 2.0f + 1.0f,
                       output1(x, y));
                return -1;
            }
            // Compare with device 2 output
            if (input(x, y) * 2.0f + 1.0f != output2(x, y)) {
                printf("Error at (%d, %d): %f != %f\n", x, y, input(x, y) * 2.0f + 1.0f,
                       output2(x, y));
                return -1;
            }
        }
    }
    
    // We need to free our GPU buffers before destroying the contexts.
    input.device_free();
    output1.device_free();
    output2.device_free();
    
    // Free the context we created.
    destroy_context();
    
    printf("Success!\n");
    return 0;
}

#endif
