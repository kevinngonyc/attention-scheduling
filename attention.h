#include "workload.h"

struct QKV;

float runPrecomputedAttention(QKV* m, const std::vector<Tile>& tiles, const std::vector<int>& schedule);
int maxBlocks();