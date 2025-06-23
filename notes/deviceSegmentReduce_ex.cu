#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm> // For std::for_each
#include <limits>    // For std::numeric_limits

// Required for CUB library functions
#include <cub/cub.cuh>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    //
    // Host-side data setup
    //
    // Input data array. This data will be segmented.
    // Example: {1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {10}
    std::vector<int> h_in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    size_t num_items = h_in.size();

    // Segment offsets. These define the start of each segment.
    // The segments are:
    // [h_in[0]...h_in[2]] (min: 1)
    // [h_in[3]...h_in[4]] (min: 4)
    // [h_in[5]...h_in[8]] (min: 6)
    // [h_in[9]...h_in[9]] (min: 10)
    std::vector<int> h_offsets = {0, 3, 5, 9};
    size_t num_segments = h_offsets.size();

    // Expected output minimums for verification
    std::vector<int> h_expected_out = {1, 4, 6, 10};

    std::cout << "Input array: ";
    std::for_each(h_in.begin(), h_in.end(), [](int x){ std::cout << x << " "; });
    std::cout << std::endl;

    std::cout << "Segment offsets: ";
    std::for_each(h_offsets.begin(), h_offsets.end(), [](int x){ std::cout << x << " "; });
    std::cout << std::endl;

    //
    // Device-side data allocation and copy
    //
    int *d_in = nullptr;
    int *d_offsets = nullptr;
    int *d_out = nullptr; // Device buffer for segment minimums
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate input array on device
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(int) * num_items));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), sizeof(int) * num_items, cudaMemcpyHostToDevice));

    // Allocate offsets array on device
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * num_segments));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), sizeof(int) * num_segments, cudaMemcpyHostToDevice));

    // Allocate output array on device
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * num_segments));

    //
    // Determine temporary device storage requirements for Min operation
    //
    // The cub::DeviceSegmentedReduce::Min function requires a temporary storage buffer.
    // The first call to Min (with d_temp_storage = nullptr) determines the size needed.
    CUDA_CHECK(cub::DeviceSegmentedReduce::Min(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        d_offsets,
        d_offsets + 1, // End offsets (inclusive). The (offsets + 1) defines the exclusive end of each segment.
        num_segments
    ));

    // Allocate temporary storage on device
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    //
    // Perform segmented reduction (Min operation)
    //
    // Now, call cub::DeviceSegmentedReduce::Min with the allocated temporary storage.
    // This performs a minimum reduction on each segment defined by the offsets.
    CUDA_CHECK(cub::DeviceSegmentedReduce::Min(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        d_offsets,
        d_offsets + 1, // End offsets. Note: CUB expects an array of begin offsets and an array of end offsets.
        num_segments
    ));

    //
    // Copy results back to host and print
    //
    std::vector<int> h_out(num_segments);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(int) * num_segments, cudaMemcpyDeviceToHost));

    std::cout << "\nSegment minimums (actual): ";
    bool all_match = true;
    for (size_t i = 0; i < num_segments; ++i) {
        std::cout << h_out[i] << " ";
        if (h_out[i] != h_expected_out[i]) {
            all_match = false;
        }
    }
    std::cout << std::endl;

    std::cout << "Segment minimums (expected): ";
    std::for_each(h_expected_out.begin(), h_expected_out.end(), [](int x){ std::cout << x << " "; });
    std::cout << std::endl;

    if (all_match) {
        std::cout << "\nAll segment minimums match expected values! Example successful." << std::endl;
    } else {
        std::cout << "\nSome segment minimums do not match expected values. There might be an issue." << std::endl;
    }

    //
    // Cleanup device memory
    //
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(d_out);
    CUDA_CHECK(cudaFree(d_temp_storage));

    return 0;
}
