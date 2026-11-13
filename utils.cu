#include <string>
#include <vector>

void printMatrix(const float* matrix, int stride, int rows, int cols) {
    std::vector<float> h(rows * stride);
    cudaMemcpy(h.data(), matrix, rows * stride * sizeof(float), cudaMemcpyDeviceToHost);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++)
            printf("%7.3f ", h[r * stride + c]);
        printf("\n");
    }
}

void saveData(const std::string& dir, const std::string& name, const void* data, size_t bytes) {
    std::vector<char> h(bytes);
    cudaMemcpy(h.data(), data, bytes, cudaMemcpyDeviceToHost);
    std::string path = dir + "/" + name;
    FILE* f = fopen(path.c_str(), "wb");
    fwrite(h.data(), 1, bytes, f);
    fclose(f);
}

void check(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}