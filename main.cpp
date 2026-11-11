#include "generate.h"
#include "workload.h"

int main() {
    int d_k = 64;
    int d_v = 64;

    std::vector<int> seqlens = {3, 5, 2};
    std::vector<int> cu_seqlens = makeCuSeqlens(seqlens);

    QKV* m = createQKV(cu_seqlens.data(), (int)seqlens.size(), d_k, d_v);
    printQKV(m, 4, 4);
    freeQKV(m);
    saveQKV(m, "data");
}