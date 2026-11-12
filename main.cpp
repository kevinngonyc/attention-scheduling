#include "generate.h"
#include "workload.h"
#include "attention.h"

int main() {
    int d_k = 64;
    int d_v = 64;

    std::vector<int> seqlens = {3, 5, 2};
    std::vector<int> cu_seqlens = makeCuSeqlens(seqlens);

    QKV* m = createQKV(cu_seqlens.data(), (int)seqlens.size(), d_k, d_v);

    std::vector<Tile> tiles = makeTiles(cu_seqlens);
    runAttention(m, tiles);

    printQKV(m, 4, 4);
    saveQKV(m, "data");

    freeQKV(m);   
}