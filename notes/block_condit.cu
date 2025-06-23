// This CUDA kernel checks if any thread in a block passes a condition.
// If the condition is met by at least one thread, it sets a corresponding
// element in a global output array to 1 for that block.

// Global function (kernel) that runs on the GPU.
// __global__ indicates that this function can be called from the host (CPU)
// and executed on the device (GPU).
__global__ void checkConditionPerBlock(int* inputData, int* outputArray, int N) {
    // blockIdx.x: The X-dimension index of the current block within the grid.
    // blockDim.x: The X-dimension size of each block (number of threads per block).
    // threadIdx.x: The X-dimension index of the current thread within its block.

    // Calculate the global thread ID.
    // This identifies each unique thread across the entire grid.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Declare a shared memory variable within the block.
    // __shared__ means this variable is allocated in the shared memory of the block,
    // accessible by all threads within the same block.
    // It's initialized to 0 (false) by default.
    __shared__ bool conditionMetInBlock;

    // The first thread in the block (threadIdx.x == 0) initializes the shared variable.
    // This ensures it's set to false at the beginning of each block's execution.
    if (threadIdx.x == 0) {
        conditionMetInBlock = false;
    }

    // Synchronize all threads in the block.
    // This ensures that 'conditionMetInBlock' is initialized before any thread
    // attempts to write to it or read from it.
    __syncthreads();

    // Check if the current thread's data meets the condition.
    // In this example, the condition is `inputData[globalThreadId] > 100`.
    if (globalThreadId < N && inputData[globalThreadId] > 100) {
        // If the condition is met, use an atomic operation to set `conditionMetInBlock` to true.
        // atomicExch: Atomically exchanges the value at a memory location with a new value.
        // Here, it sets conditionMetInBlock to true (1) if it was false (0), ensuring
        // only one thread "wins" the exchange and sets it, but all threads will see the true value.
        // It's more robust than a simple assignment for race conditions if multiple threads
        // might meet the condition simultaneously and attempt to write.
        // However, given the `__syncthreads()` before the read in the next step,
        // a simple `conditionMetInBlock = true;` would also work effectively here
        // as long as the intention is just to set it true if any thread finds it.
        // Using atomicExch here is a good practice for demonstrating atomicity,
        // but for a simple boolean flag where any thread setting it to true is sufficient,
        // it might be overkill if subsequent reads are guarded by __syncthreads().
        atomicExch((int*)&conditionMetInBlock, 1); // Set to true (1)
    }

    // Synchronize all threads in the block again.
    // This ensures that all threads have completed their condition check and any
    // atomic operations on `conditionMetInBlock` are visible to all.
    __syncthreads();

    // The first thread in the block is responsible for writing the result to global memory.
    // This prevents multiple threads from writing to the same location in `outputArray`
    // for the same block, avoiding race conditions on global memory.
    if (threadIdx.x == 0) {
        // blockIdx.x maps directly to the index in `outputArray` representing this block.
        outputArray[blockIdx.x] = conditionMetInBlock ? 1 : 0;
    }
}

// Example Host (CPU) Code to launch the kernel (for context):
/*
#include <iostream>
#include <vector>

// Forward declaration of the kernel
__global__ void checkConditionPerBlock(int* inputData, int* outputArray, int N);

int main() {
    const int N = 1000; // Total number of elements
    const int blockSize = 256; // Number of threads per block
    // Calculate the number of blocks needed. Ceil division.
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // Host (CPU) data
    std::vector<int> h_inputData(N);
    // Initialize input data (e.g., some values meeting the condition, some not)
    for (int i = 0; i < N; ++i) {
        if (i % 50 == 0) { // Every 50th element meets the condition
            h_inputData[i] = 101 + i;
        } else {
            h_inputData[i] = i % 100;
        }
    }

    // Device (GPU) pointers
    int* d_inputData;
    int* d_outputArray; // One element per block

    // Allocate memory on the device
    cudaMalloc(&d_inputData, N * sizeof(int));
    cudaMalloc(&d_outputArray, numBlocks * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_inputData, h_inputData.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize output array on device to all zeros (optional, but good practice)
    cudaMemset(d_outputArray, 0, numBlocks * sizeof(int));

    // Launch the kernel
    // The first two arguments are the grid dimensions (blocks per grid)
    // and block dimensions (threads per block).
    checkConditionPerBlock<<<numBlocks, blockSize>>>(d_inputData, d_outputArray, N);

    // Wait for the kernel to complete
    cudaDeviceSynchronize();

    // Copy the result back from device to host
    std::vector<int> h_outputArray(numBlocks);
    cudaMemcpy(h_outputArray.data(), d_outputArray, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "Results per block (1 if condition met, 0 otherwise):" << std::endl;
    for (int i = 0; i < numBlocks; ++i) {
        std::cout << "Block " << i << ": " << h_outputArray[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_inputData);
    cudaFree(d_outputArray);

    return 0;
}
*/
