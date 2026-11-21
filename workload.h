#pragma once
#include <vector>
#include <string>
#include "config.h"

struct Tile {
    int q_start, q_end;
    int kv_start, kv_end;
    int cost() const { 
        return (kv_end - kv_start + TILE_ROWS - 1) / TILE_ROWS; 
    }
    bool operator<(const Tile& other) const {
        return cost() < other.cost();
    }
};

std::vector<Tile> makeTiles(const std::vector<int>& cu_seqlens);
std::vector<int> makeCuSeqlens(const std::vector<int>& seqlens);
std::vector<std::vector<int>> readWorkloads(const std::string& path);