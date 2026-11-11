#include <cuda_fp16.h>
#define THREADS_PER_BLOCK 256

__global__ void fill(__half* p, int n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned x = (unsigned)(i + offset) * 2654435761u;
        p[i] = __float2half((x >> 16) / 32768.0f - 1.0f);
    }
}

void cudaFill(__half* p, int n, int offset) {
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill<<<blocks, THREADS_PER_BLOCK>>>(p, n, offset);
}