#include <cuda_fp16.h>
#include "config.h"
#include "workload.h"
#include "qkv.cuh"

__global__ void attention(
    const __half* Q, 
    const __half* K, 
    const __half* V,
    __half* O, 
    const Tile* tiles
) {
    Tile t = tiles[blockIdx.x];
    int row = t.q_start + threadIdx.x;
    if (row >= t.q_end) return;

    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    float q[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++)
        q[d] = __half2float(Q[row * HEAD_DIM + d]) * scale;

    float max_score = -INFINITY;
    float weight_sum = 0.0f;
    float out[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) out[d] = 0.0f;

    for (int i = t.kv_start; i < t.kv_end; i++) {
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++)
            score += q[d] * __half2float(K[i * HEAD_DIM + d]);

        float new_max = fmaxf(max_score, score);
        float correction = expf(max_score - new_max);
        float weight = expf(score - new_max);
        weight_sum = weight_sum * correction + weight;
        for (int d = 0; d < HEAD_DIM; d++)
            out[d] = out[d] * correction + weight * __half2float(V[i * HEAD_DIM + d]);
        max_score = new_max;
    }

    for (int d = 0; d < HEAD_DIM; d++)
        O[row * HEAD_DIM + d] = __float2half(out[d] / weight_sum);
}

void runAttention(QKV* m, const std::vector<Tile>& tiles) {
    Tile* tile_data;
    size_t bytes = tiles.size() * sizeof(Tile);
    cudaMalloc(&tile_data, bytes);
    cudaMemcpy(tile_data, tiles.data(), bytes, cudaMemcpyHostToDevice);

    // One block per tile, one thread per row
    attention<<<tiles.size(), TILE_ROWS>>>(m->q, m->k, m->v, m->o, tile_data);
    cudaDeviceSynchronize();

    cudaFree(tile_data);
}