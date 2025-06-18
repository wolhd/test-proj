#include <iostream>     // For input/output operations (std::cout, std::cerr)
#include <vector>       // For std::vector to manage host-side arrays
#include <algorithm>    // For std::sort (CPU reference)
#include <random>       // For std::mt19937, std::uniform_real_distribution for random data generation
#include <chrono>       // For std::chrono to measure execution time
#include <cmath>        // For std::abs (for floating-point comparison)

// CUB Headers
#include <cub/cub.cuh>              // General CUB utilities
// Include for DeviceMergeSort instead of DeviceSort
#include <cub/device/device_merge_sort.cuh> // DeviceMergeSort for sorting operations

// Macro for checking CUDA errors. This helps in debugging by printing error messages
// and exiting the program if a CUDA API call fails.
#define cudaCheckError() {                                                                  \
    cudaError_t err = cudaGetLastError();                                                   \
    if (err != cudaSuccess) {                                                               \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                                                 \
    }                                                                                       \
}

// Macro for checking CUB errors. CUB functions often return a cub::Result.
#define CUB_CHECK(call) {                                                                   \
    cub::Result error = call;                                                               \
    if (error != cub::CUB_SUCCESS) {                                                        \
        std::cerr << "CUB error at " << __FILE__ << ":" << __LINE__ << ": " << cub::CUBGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE);                                                                 \
    }                                                                                       \
}

int main() {
    // Define the size of the array. Using a large number to demonstrate GPU's advantage.
    const int N = 4000000; // 4 million elements

    std::cout << "--- CUDA CUB DeviceMergeSort Example ---" << std::endl;
    std::cout << "Array Size (N): " << N << std::endl;

    // 1. Host (CPU) Data Generation
    // Create a host vector to store the input data.
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);     // To store sorted data from GPU
    std::vector<float> h_input_cpu_copy(N); // To store a copy for CPU reference sort

    // Initialize random number generator.
    std::random_device rd;           // Seeds the random number generator
    std::mt19937 gen(rd());          // Mersenne Twister engine
    // Distribution for floating-point numbers between 0.0 and 1000.0
    std::uniform_real_distribution<> dis(0.0, 1000.0);

    // Populate the host vector with random float values.
    auto start_data_gen = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        h_input[i] = dis(gen);
    }
    auto end_data_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> data_gen_time = end_data_gen - start_data_gen;
    std::cout << "Host data generation complete. Time: " << data_gen_time.count() << " seconds." << std::endl;

    // Create a copy for CPU-side verification.
    h_input_cpu_copy = h_input;

    // 2. Device (GPU) Memory Allocation
    float *d_input = nullptr;   // Pointer for input array on device
    float *d_output = nullptr;  // Pointer for output array on device (sorted result)

    // Allocate memory on the GPU for input and output arrays.
    cudaMalloc(&d_input, N * sizeof(float));
    cudaCheckError();
    cudaMalloc(&d_output, N * sizeof(float));
    cudaCheckError();
    std::cout << "Device memory allocated." << std::endl;

    // 3. Copy input data from Host to Device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();
    std::cout << "Input data copied from host to device." << std::endl;

    // 4. Determine temporary device storage requirements for CUB DeviceMergeSort
    // CUB functions often require a temporary buffer for their internal operations.
    // We first query the size needed for this buffer.
    size_t temp_storage_bytes = 0;
    // The first call to Sort (with temp_storage_bytes = 0) determines the buffer size.
    // This is a "dry run" to get the required memory.
    CUB_CHECK(cub::DeviceMergeSort::Sort(
        nullptr,            // d_temp_storage: pointer to the temporary storage allocation, or null to query its size
        temp_storage_bytes, // temp_storage_bytes: output variable for the size of the temporary storage
        d_input,            // d_keys_in: input keys
        d_output,           // d_keys_out: sorted keys output
        N                   // num_items: number of items to sort
    ));

    // 5. Allocate temporary device storage
    void *d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cudaCheckError();
    std::cout << "Temporary device storage allocated: "
              << static_cast<double>(temp_storage_bytes) / (1024.0 * 1024.0) << " MB." << std::endl;

    // 6. Perform sorting on the GPU using CUB DeviceMergeSort
    std::cout << "Sorting data on GPU using CUB DeviceMergeSort..." << std::endl;
    auto start_gpu_sort = std::chrono::high_resolution_clock::now();

    // Call Sort to perform the merge sort.
    // The keys are read from d_input and the sorted keys are written to d_output.
    CUB_CHECK(cub::DeviceMergeSort::Sort(
        d_temp_storage,     // d_temp_storage: pointer to the temporary storage allocation
        temp_storage_bytes, // temp_storage_bytes: size of the temporary storage allocation
        d_input,            // d_keys_in: input keys (unsorted)
        d_output,           // d_keys_out: sorted keys output
        N                   // num_items: number of items to sort
    ));
    cudaCheckError(); // Check for any CUDA errors during sort operation (e.g., kernel launch issues)
    // Synchronize the device to ensure sorting is complete before measuring time.
    cudaDeviceSynchronize();
    cudaCheckError();
    auto end_gpu_sort = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_sort_time = end_gpu_sort - start_gpu_sort;
    std::cout << "GPU sorting complete. Time: " << gpu_sort_time.count() << " seconds." << std::endl;

    // 7. Copy sorted results from Device to Host
    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    std::cout << "Sorted results copied from device to host." << std::endl;

    // 8. CPU Reference Calculation for Verification
    std::cout << "\nPerforming CPU reference calculation for verification..." << std::endl;
    auto start_cpu_sort = std::chrono::high_resolution_clock::now();
    std::sort(h_input_cpu_copy.begin(), h_input_cpu_copy.end()); // Standard C++ sort on CPU
    auto end_cpu_sort = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_sort_time = end_cpu_sort - start_cpu_sort;
    std::cout << "CPU sorting complete. Time: " << cpu_sort_time.count() << " seconds." << std::endl;

    // 9. Verification of results
    bool success = true;
    float tolerance = 1e-5f; // Tolerance for floating point comparisons due to precision differences
    int mismatch_count = 0;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_output[i] - h_input_cpu_copy[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": CUDA=" << h_output[i]
                      << ", CPU=" << h_input_cpu_copy[i] << ", Diff=" << std::abs(h_output[i] - h_input_cpu_copy[i]) << std::endl;
            success = false;
            mismatch_count++;
            if (mismatch_count > 10) { // Limit mismatch output to prevent excessive console spam
                std::cerr << "Too many mismatches, stopping detailed output." << std::endl;
                break;
            }
        }
    }

    if (success) {
        std::cout << "\nVerification successful! CUDA CUB results match CPU results within tolerance." << std::endl;
    } else {
        std::cerr << "\nVerification FAILED! Total mismatches: " << mismatch_count << std::endl;
    }

    // 10. Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage); // Remember to free the temporary storage
    std::cout << "Device memory freed." << std::endl;

    return 0;
}
