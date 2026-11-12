#include "generate.h"
#include "workload.h"
#include "attention.h"

int main(int argc, char** argv) {
    std::string path;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-i" && i + 1 < argc)
            path = argv[++i];
    }
    if (path.empty()) {
        fprintf(stderr, "Usage: %s -i FILE\n", argv[0]);
        return 1;
    }

    std::vector<std::vector<int>> workloads = readWorkloads(path);

    for (size_t w = 0; w < workloads.size(); w++) {
        const std::vector<int>& seqlens = workloads[w];
        printf("== Workload %zu: %zu sequences\n", w, seqlens.size());

        std::vector<int> cu_seqlens = makeCuSeqlens(seqlens);
        QKV* m = createQKV(cu_seqlens.data(), (int)seqlens.size());

        std::vector<Tile> tiles = makeTiles(cu_seqlens);
        runAttention(m, tiles);

        printQKV(m, 4, 4);
        saveQKV(m, "data/workload_" + std::to_string(w));
        freeQKV(m);
    }
}