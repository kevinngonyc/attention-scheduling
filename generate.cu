#include <cstdio>
#include <vector>
#include <cuda_fp16.h>
#include <string>
#include "generate.h"

#define THREADS_PER_BLOCK 256

struct QKV {
    __half *q, *k, *v;
    int *cu_seqlens;
    int num_seqs;
    int tokens;
    int d_k;
    int d_v;
};

__global__ void fill(__half* p, int n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned x = (unsigned)(i + offset) * 2654435761u;
        p[i] = __float2half((x >> 16) / 32768.0f - 1.0f);
    }
}

QKV* createQKV(const int* cu_seqlens, int num_seqs, int d_k, int d_v) {
    QKV* m = new QKV;
    m->num_seqs = num_seqs;
    m->tokens = cu_seqlens[num_seqs];
    m->d_k = d_k;
    m->d_v = d_v;

    cudaMalloc(&m->cu_seqlens, (num_seqs + 1) * sizeof(int));
    cudaMemcpy(m->cu_seqlens, cu_seqlens, (num_seqs + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int n_qk = m->tokens * d_k;
    int n_v  = m->tokens * d_v;

    cudaMalloc(&m->q, n_qk * sizeof(__half));
    cudaMalloc(&m->k, n_qk * sizeof(__half));
    cudaMalloc(&m->v, n_v  * sizeof(__half));

    fill<<<(n_qk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(m->q, n_qk, 0);
    fill<<<(n_qk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(m->k, n_qk, n_qk);
    fill<<<(n_v  + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(m->v, n_v,  2 * n_qk);
    cudaDeviceSynchronize();

    return m;
}

static void printMatrix(const __half* data, int stride, int rows, int cols) {
    std::vector<__half> h(rows * stride);
    cudaMemcpy(h.data(), data, rows * stride * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++)
            printf("%7.3f ", __half2float(h[r * stride + c]));
        printf("\n");
    }
}

void printQKV(const QKV* m, int rows, int cols) {
    printf("Q:\n"); printMatrix(m->q, m->d_k, rows, std::min(cols, m->d_k));
    printf("K:\n"); printMatrix(m->k, m->d_k, rows, std::min(cols, m->d_k));
    printf("V:\n"); printMatrix(m->v, m->d_v, rows, std::min(cols, m->d_v));
}

void freeQKV(QKV* m) {
    cudaFree(m->q); cudaFree(m->k); cudaFree(m->v); cudaFree(m->cu_seqlens);
    delete m;
}

static void saveData(const std::string& dir, const std::string& name, const void* data, size_t bytes) {
    std::vector<char> h(bytes);
    cudaMemcpy(h.data(), data, bytes, cudaMemcpyDeviceToHost);
    std::string path = dir + "/" + name;
    FILE* f = fopen(path.c_str(), "wb");
    fwrite(h.data(), 1, bytes, f);
    fclose(f);
}

void saveQKV(const QKV* m, const std::string& dir) {
    saveData(dir, "q.bin", m->q, (size_t)m->tokens * m->d_k * sizeof(__half));
    saveData(dir, "k.bin", m->k, (size_t)m->tokens * m->d_k * sizeof(__half));
    saveData(dir, "v.bin", m->v, (size_t)m->tokens * m->d_v * sizeof(__half));
    saveData(dir, "cu_seqlens.bin", m->cu_seqlens, (m->num_seqs + 1) * sizeof(int));
}