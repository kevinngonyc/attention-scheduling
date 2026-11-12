#include <fstream>
#include <sstream>
#include "workload.h"
#include "config.h"

std::vector<Tile> makeTiles(const std::vector<int>& cu_seqlens) {
    std::vector<Tile> tiles;
    int num_seqs = cu_seqlens.size();
    for (int s = 0; s < num_seqs; s++) {
        int seq_start = cu_seqlens[s], seq_end = cu_seqlens[s + 1];
        for (int r = seq_start; r < seq_end; r += TILE_ROWS) {
            tiles.push_back({r, std::min(r + TILE_ROWS, seq_end), seq_start, seq_end});
        }
    }
    return tiles;
}

std::vector<int> makeCuSeqlens(const std::vector<int>& seqlens) {
    std::vector<int> cu = {0};
    for (int len : seqlens) cu.push_back(cu.back() + len);
    return cu;
}

std::vector<std::vector<int>> readWorkloads(const std::string& path) {
    std::ifstream f(path);
    std::vector<std::vector<int>> workloads;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream nums(line);
        std::vector<int> seqlens;
        int len;
        while (nums >> len) seqlens.push_back(len);
        workloads.push_back(seqlens);
    }
    return workloads;
}