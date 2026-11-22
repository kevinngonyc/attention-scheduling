#include <string>
#include "config.h"
#include "workload.h"
#include "utils.h"
#include "qkv.cuh"
#define RUNS 100

struct SMInfo {
    int block, smid;
    unsigned long long start, end;
};

__device__ void processTile(
    const float* Q, 
    const float* K, 
    const float* V,
    float* O, 
    const Tile& t,
    float* K_shared,
    float* V_shared
) {
    int row = t.q_start + threadIdx.x;
    bool active = row < t.q_end;

    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    float q[HEAD_DIM];
    if (active) 
        for (int d = 0; d < HEAD_DIM; d++)
            q[d] = Q[row * HEAD_DIM + d] * scale;

    float max_score = -INFINITY;
    float weight_sum = 0.0f;
    float out[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) out[d] = 0.0f;

    for (int kv = t.kv_start; kv < t.kv_end; kv += TILE_ROWS) {
        int n = min(TILE_ROWS, t.kv_end - kv);

        for (int i = threadIdx.x; i < n * HEAD_DIM; i += TILE_ROWS) {
            K_shared[i] = K[kv * HEAD_DIM + i];
            V_shared[i] = V[kv * HEAD_DIM + i];
        }
        __syncthreads();

        if (active) {
            for (int j = 0; j < n; j++) {
                float score = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++)
                    score += q[d] * K_shared[j * HEAD_DIM + d];

                float new_max = fmaxf(max_score, score);
                float correction = expf(max_score - new_max);
                float weight = expf(score - new_max);
                weight_sum = weight_sum * correction + weight;
                for (int d = 0; d < HEAD_DIM; d++)
                    out[d] = out[d] * correction + weight * V_shared[j * HEAD_DIM + d];
                max_score = new_max;
            }
        }
        __syncthreads();
    }

    if (active) 
        for (int d = 0; d < HEAD_DIM; d++)
            O[row * HEAD_DIM + d] = out[d] / weight_sum;
}

__device__ __forceinline__ int smid() {
    int id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

__device__ __forceinline__ unsigned long long globalTimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void precomputedAttention(
    const float* Q, 
    const float* K, 
    const float* V,
    float* O, 
    const Tile* tiles,
    const int* schedule,
    SMInfo* sm_info = nullptr
) {
    __shared__ float K_shared[TILE_ROWS * HEAD_DIM];
    __shared__ float V_shared[TILE_ROWS * HEAD_DIM];

    for (int i = schedule[blockIdx.x]; i < schedule[blockIdx.x + 1]; i++) {
        Tile t = tiles[i];
        unsigned long long start = (sm_info != nullptr) ? globalTimer() : 0;
        processTile(Q, K, V, O, t, K_shared, V_shared);
        if (sm_info != nullptr) {
            __syncthreads();
            if (threadIdx.x == 0)
                sm_info[i] = {(int)blockIdx.x, smid(), start, globalTimer()};
        }
    }
}

void runSMInfoAttention(QKV* m, const std::vector<Tile>& tiles, const std::vector<int>& schedule, const std::string& path) {
    Tile* tile_data;
    size_t tile_bytes = tiles.size() * sizeof(Tile);
    cudaMalloc(&tile_data, tile_bytes);
    cudaMemcpy(tile_data, tiles.data(), tile_bytes, cudaMemcpyHostToDevice);

    int* schedule_data;
    size_t schedule_bytes = schedule.size() * sizeof(int);
    cudaMalloc(&schedule_data, schedule_bytes);
    cudaMemcpy(schedule_data, schedule.data(), schedule_bytes, cudaMemcpyHostToDevice);

    int num_blocks = (int)schedule.size() - 1;
    SMInfo* sm_info_data;
    cudaMalloc(&sm_info_data, tiles.size() * sizeof(SMInfo));

    // run kernel once for GPU clock ramp-up
    precomputedAttention<<<num_blocks, TILE_ROWS>>>(m->q, m->k, m->v, m->o, tile_data, schedule_data, sm_info_data);
    cudaDeviceSynchronize();

    precomputedAttention<<<num_blocks, TILE_ROWS>>>(m->q, m->k, m->v, m->o, tile_data, schedule_data, sm_info_data);
    cudaDeviceSynchronize();
    
    std::vector<SMInfo> sm_info(tiles.size());
    cudaMemcpy(sm_info.data(), sm_info_data, sm_info.size() * sizeof(SMInfo), cudaMemcpyDeviceToHost);

    FILE* f = fopen(path.c_str(), "w");
    fprintf(f, "tile,block,smid,q_start,q_end,kv_start,kv_end,cost,start_ns,end_ns\n");
    for (size_t i = 0; i < tiles.size(); i++) {
        const Tile& t = tiles[i];
        fprintf(f, "%zu,%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n", i, sm_info[i].block, sm_info[i].smid,
                t.q_start, t.q_end, t.kv_start, t.kv_end, t.cost(), sm_info[i].start, sm_info[i].end);
    }

    fclose(f);
    cudaFree(tile_data);
    cudaFree(schedule_data);
    cudaFree(sm_info_data);
}

float runPrecomputedAttention(QKV* m, const std::vector<Tile>& tiles, const std::vector<int>& schedule) {
    Tile* tile_data;
    size_t tile_bytes = tiles.size() * sizeof(Tile);
    cudaMalloc(&tile_data, tile_bytes);
    cudaMemcpy(tile_data, tiles.data(), tile_bytes, cudaMemcpyHostToDevice);

    int* schedule_data;
    size_t schedule_bytes = schedule.size() * sizeof(int);
    cudaMalloc(&schedule_data, schedule_bytes);
    cudaMemcpy(schedule_data, schedule.data(), schedule_bytes, cudaMemcpyHostToDevice);

    int num_blocks = (int)schedule.size() - 1;

    // run kernel once for GPU clock ramp-up
    precomputedAttention<<<num_blocks, TILE_ROWS>>>(m->q, m->k, m->v, m->o, tile_data, schedule_data);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int r = 0; r < RUNS; r++)
        precomputedAttention<<<num_blocks, TILE_ROWS>>>(m->q, m->k, m->v, m->o, tile_data, schedule_data);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(tile_data);
    cudaFree(schedule_data);
    return ms / RUNS;
}

// float runOnlineAttention(QKV* m, const std::vector<Tile>& tiles) {
//     Tile* tile_data;
//     size_t bytes = tiles.size() * sizeof(Tile);
//     cudaMalloc(&tile_data, bytes);
//     cudaMemcpy(tile_data, tiles.data(), bytes, cudaMemcpyHostToDevice);

//     // run kernel once for GPU clock ramp-up
//     onlineAttention<<<tiles.size(), TILE_ROWS>>>(m->q, m->k, m->v, m->o, tile_data);
//     cudaDeviceSynchronize();

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start);
//     for (int r = 0; r < RUNS; r++)
//         onlineAttention<<<tiles.size(), TILE_ROWS>>>(m->q, m->k, m->v, m->o, tile_data);
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     float ms;
//     cudaEventElapsedTime(&ms, start, stop);

//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     cudaFree(tile_data);
//     return ms / RUNS;
// }

int maxBlocks() {
    int dev, per_sm;
    cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, precomputedAttention, TILE_ROWS, 0);
    return prop.multiProcessorCount * per_sm;
}