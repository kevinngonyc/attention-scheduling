#include <cuda_fp16.h>

struct QKV {
    __half *q, *k, *v, *o;
    int *cu_seqlens;
    int num_seqs;
    int tokens;
    int d_k;
    int d_v;
};