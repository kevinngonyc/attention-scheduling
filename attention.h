#include "workload.h"

struct QKV;

void runAttention(QKV* m, const std::vector<Tile>& tiles);