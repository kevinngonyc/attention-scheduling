#include <filesystem>
#include <string>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "generate.h"
#include "workload.h"
#include "attention.h"
#include "inspector.h"

int main(int argc, char** argv) {
    std::string input_path = "inputs/default.txt";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-i" && i + 1 < argc)
            input_path = argv[++i];
    }

    std::string name = std::filesystem::path(input_path).stem().string();
    std::string output_dir = "data/" + name;
    std::filesystem::create_directories(output_dir);

    FILE* csv = fopen((output_dir + "/kernel_times.csv").c_str(), "w");
    fprintf(csv, "workload,num_seqs,num_tiles,imbalance,naive,lpt\n");

    std::vector<std::vector<int>> workloads = readWorkloads(input_path);

    for (size_t w = 0; w < workloads.size(); w++) {
        const std::vector<int>& seqlens = workloads[w];
        printf("\nRunning Workload %zu...\n", w);

        // imbalance: longest sequence length / mean sequence length.
        // 1.0 means all sequences are the same length; larger means a few long
        // sequences dominate. (Unlike skewness, this grows with the long/short ratio.)
        double total_len = 0, max_len = 0;
        for (int len : seqlens) {
            total_len += len;
            max_len = std::max(max_len, (double)len);
        }
        double imbalance = total_len > 0 ? max_len / (total_len / seqlens.size()) : 0;

        std::vector<int> cu_seqlens = makeCuSeqlens(seqlens);
        QKV* m = createQKV(cu_seqlens.data(), (int)seqlens.size());
        std::vector<Tile> tiles = makeTiles(cu_seqlens);

        std::vector<int> naive_schedule = naiveSchedule((int)tiles.size());
        float naive_kernel_time = runPrecomputedAttention(m, tiles, naive_schedule);
        printf("Kernel Time (Naive Schedule): %.3f\n", naive_kernel_time);

        std::vector<int> lpt_schedule = lptSchedule(tiles, maxBlocks());
        float lpt_kernel_time = runPrecomputedAttention(m, tiles, lpt_schedule);
        printf("Kernel Time (LPT Schedule): %.3f\n", lpt_kernel_time);

        fprintf(csv, "%zu,%zu,%zu,%.3f,%.3f,%.3f\n", w, seqlens.size(), tiles.size(),
                imbalance, naive_kernel_time, lpt_kernel_time);

        std::string qkv_output = output_dir + "/workload_" + std::to_string(w);
        std::filesystem::create_directories(qkv_output);
        // saveQKV(m, qkv_output);
        freeQKV(m);
    }
    fclose(csv);
}