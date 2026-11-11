#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>


void cudaFill(__half* p, int n, int offset);

void printMatrix(const __half* matrix, int rows, int cols) {
    std::vector<__half> h(rows * cols);
    cudaMemcpy(h.data(), matrix, rows * cols * sizeof(__half), cudaMemcpyDeviceToHost);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%7.3f ", __half2float(h[r * cols + c]));
        }
        printf("\n");
    }
}

int main(int argc, char** argv)
{
    int n = 1 << 20;   // placeholder size
    __half *Q, *K, *V;
    cudaMalloc(&Q, n * sizeof(__half));
    cudaMalloc(&K, n * sizeof(__half));
    cudaMalloc(&V, n * sizeof(__half));

    cudaFill(Q, n, 0);
    cudaFill(K, n, n);
    cudaFill(V, n, 2 * n);
    cudaDeviceSynchronize();

    printf("Q:\n");
    print_matrix(Q, 4, 4);
    printf("K:\n");
    print_matrix(K, 4, 4);
    printf("V:\n");
    print_matrix(V, 4, 4);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
}