#include "workload.h"

struct QKV;

float runAttention(QKV* m, const std::vector<Tile>& tiles);