#include <filesystem>
#include <string>
#include <cstdio>
#include <cmath>
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
    fprintf(csv, "workload,num_seqs,num_tiles,skew,naive,lpt\n");

    std::vector<std::vector<int>> workloads = readWorkloads(input_path);

    for (size_t w = 0; w < workloads.size(); w++) {
        const std::vector<int>& seqlens = workloads[w];
        printf("\nRunning Workload %zu...\n", w);

        // sample skewness: n / ((n-1)(n-2)) * sum(((x - mean) / s)^3)
        double n = seqlens.size();
        double mean = 0;
        for (int x : seqlens) mean += x;
        mean /= n;

        double var = 0;
        for (int x : seqlens) var += (x - mean) * (x - mean);
        double sd = sqrt(var / (n - 1));

        double cubes = 0;
        for (int x : seqlens) cubes += pow((x - mean) / sd, 3);

        double skew = 0;
        if (n >= 3 && sd > 0)
            skew = n / ((n - 1) * (n - 2)) * cubes;

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
                skew, naive_kernel_time, lpt_kernel_time);

        std::string qkv_output = output_dir + "/workload_" + std::to_string(w);
        std::filesystem::create_directories(qkv_output);
        // saveQKV(m, qkv_output);
        freeQKV(m);
    }
    fclose(csv);
}