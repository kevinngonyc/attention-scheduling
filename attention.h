#include <string>
#include "workload.h"

struct QKV;

float runPrecomputedAttention(QKV* m, const std::vector<Tile>& tiles, const std::vector<int>& schedule);
void runSMInfoAttention(QKV* m, const std::vector<Tile>& tiles, const std::vector<int>& schedule, const std::string& path);
int maxBlocks();