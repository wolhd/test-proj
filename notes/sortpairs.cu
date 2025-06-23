#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm> // For std::is_sorted

// Include the CUB header for DeviceRadixSort
#include <cub/cub.cuh>

// Kernel to print device arrays (for debugging/verification)
template <typename T>
__global__ void PrintArrayKernel(const T* arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        printf("%d: %d\n", idx, arr[idx]);
    }
}

int main() {
    // Declare, allocate, and initialize host arrays
    int num_items = 10;

    // Keys to be sorted
    std::vector<int> h_keys_in(num_items);
    // Values associated with the keys, will be reordered along with keys
    std::vector<int> h_values_in(num_items);

    // Initialize keys with some unsorted values
    h_keys_in[0] = 50; h_values_in[0] = 500;
    h_keys_in[1] = 20; h_values_in[1] = 200;
    h_keys_in[2] = 80; h_values_in[2] = 800;
    h_keys_in[3] = 10; h_values_in[3] = 100;
    h_keys_in[4] = 70; h_values_in[4] = 700;
    h_keys_in[5] = 40; h_values_in[5] = 400;
    h_keys_in[6] = 90; h_values_in[6] = 900;
    h_keys_in[7] = 30; h_values_in[7] = 300;
    h_keys_in[8] = 60; h_values_in[8] = 600;
    h_keys_in[9] = 0;  h_values_in[9] = 0;

    std::cout << "Original Keys: ";
    for (int i = 0; i < num_items; ++i) {
        std::cout << h_keys_in[i] << (i == num_items - 1 ? "" : ", ");
    }
    std::cout << std::endl;

    std::cout << "Original Values: ";
    for (int i = 0; i < num_items; ++i) {
        std::cout << h_values_in[i] << (i == num_items - 1 ? "" : ", ");
    }
    std::cout << std::endl;

    // Declare device pointers
    int *d_keys_in = nullptr;
    int *d_keys_out = nullptr;
    int *d_values_in = nullptr;
    int *d_values_out = nullptr;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate device memory
    cudaMalloc(&d_keys_in, num_items * sizeof(int));
    cudaMalloc(&d_keys_out, num_items * sizeof(int));
    cudaMalloc(&d_values_in, num_items * sizeof(int));
    cudaMalloc(&d_values_out, num_items * sizeof(int));

    // Copy host input arrays to device
    cudaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, h_values_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);

    // Determine temporary device storage requirements
    // The first call to SortPairs with a null d_temp_storage argument
    // calculates the required size of temporary storage.
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        num_items
    );

    // Allocate temporary device storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform sort
    // Note: CUB sorts in-place if d_keys_in == d_keys_out.
    // For this example, we use separate input/output buffers for clarity.
    // The sort is stable and ascending by default.
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        num_items
    );

    // Copy sorted output arrays back to host
    std::vector<int> h_keys_out(num_items);
    std::vector<int> h_values_out(num_items);
    cudaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values_out.data(), d_values_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);

    // Print sorted output
    std::cout << "\nSorted Keys: ";
    for (int i = 0; i < num_items; ++i) {
        std::cout << h_keys_out[i] << (i == num_items - 1 ? "" : ", ");
    }
    std::cout << std::endl;

    std::cout << "Sorted Values: ";
    for (int i = 0; i < num_items; ++i) {
        std::cout << h_values_out[i] << (i == num_items - 1 ? "" : ", ");
    }
    std::cout << std::endl;

    // Verify the sort
    bool keys_sorted = std::is_sorted(h_keys_out.begin(), h_keys_out.end());
    std::cout << "\nKeys are sorted: " << (keys_sorted ? "True" : "False") << std::endl;

    // A simple check for value association (more robust checks would compare with a ground truth)
    // Here we just check if 0 key maps to 0 value, 10 to 100, etc.
    bool values_correctly_associated = true;
    for (int i = 0; i < num_items; ++i) {
        if (h_keys_out[i] * 10 != h_values_out[i]) {
            values_correctly_associated = false;
            break;
        }
    }
    std::cout << "Values are correctly associated: " << (values_correctly_associated ? "True" : "False") << std::endl;


    // Cleanup
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_temp_storage);

    return 0;
}

