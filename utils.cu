#include <string>
#include <vector>
#include <cuda_fp16.h>

void printMatrix(const __half* data, int stride, int rows, int cols) {
    std::vector<__half> h(rows * stride);
    cudaMemcpy(h.data(), data, rows * stride * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++)
            printf("%7.3f ", __half2float(h[r * stride + c]));
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