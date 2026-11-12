#include <filesystem>
#include "generate.h"
#include "workload.h"
#include "attention.h"

int main(int argc, char** argv) {
    std::string input_path = "inputs/default.txt";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-i" && i + 1 < argc)
            input_path = argv[++i];
    }

    std::string name = std::filesystem::path(input_path).stem().string();
    std::string output_dir = "data/" + name;

    std::vector<std::vector<int>> workloads = readWorkloads(input_path);

    for (size_t w = 0; w < workloads.size(); w++) {
        const std::vector<int>& seqlens = workloads[w];
        printf("\nRunning Workload %zu...\n", w);

        std::vector<int> cu_seqlens = makeCuSeqlens(seqlens);
        QKV* m = createQKV(cu_seqlens.data(), (int)seqlens.size());

        std::vector<Tile> tiles = makeTiles(cu_seqlens);
        float kernel_time = runAttention(m, tiles);
        printf("Kernel Time: %.3f\n", kernel_time);

        std::string qkv_output = output_dir + "/workload_" + std::to_string(w);
        std::filesystem::create_directories(qkv_output);
        saveQKV(m, qkv_output);
        freeQKV(m);
    }
}