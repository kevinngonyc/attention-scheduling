#include <vector>
#include <numeric>
#include <algorithm>
#include <queue>
#include <functional>
#include <utility>
#include "workload.h"

std::vector<int> naiveSchedule(int num_tiles) {
    std::vector<int> schedule(num_tiles + 1); 
    std::iota(schedule.begin(), schedule.end(), 0); 
    return schedule;
}

std::vector<int> lptSchedule(std::vector<Tile>& tiles, int num_blocks) {
    std::sort(tiles.rbegin(), tiles.rend());

    std::vector<std::vector<Tile>> blocks(num_blocks);
    using Entry = std::pair<int, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> min_pq;
    for (int b = 0; b < num_blocks; ++b) min_pq.push({0, b});

    for (const Tile& t : tiles) {
        auto [cost, b] = min_pq.top();
        min_pq.pop();
        blocks[b].push_back(t);
        min_pq.push({cost + t.cost(), b});
    }

    std::vector<Tile> flat;
    flat.reserve(tiles.size());
    std::vector<int> schedule = {0};
    for (const auto& block : blocks) {
        flat.insert(flat.end(), block.begin(), block.end());
        schedule.push_back(schedule.back() + (int)block.size());
    }
    tiles = std::move(flat);
    return schedule;
}