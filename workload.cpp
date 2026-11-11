#include "workload.h"

std::vector<int> makeCuSeqlens(const std::vector<int>& seqlens) {
    std::vector<int> cu = {0};
    for (int len : seqlens) cu.push_back(cu.back() + len);
    return cu;
}