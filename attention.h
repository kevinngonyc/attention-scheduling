#include "workload.h"
#include "qkv.cuh"

void runAttention(QKV* m, const std::vector<Tile>& tiles);