#pragma once
#include <vector>
#include <string>

struct Tile {
    int q_start, q_end;
    int kv_start, kv_end;
};

std::vector<Tile> makeTiles(const std::vector<int>& cu_seqlens);
std::vector<int> makeCuSeqlens(const std::vector<int>& seqlens);
std::vector<std::vector<int>> readWorkloads(const std::string& path);