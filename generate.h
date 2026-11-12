#include <string>
#include "qkv.cuh"

QKV* createQKV(const int* cu_seqlens, int num_seqs, int d_k, int d_v);
void printQKV(const QKV* m, int rows, int cols);
void freeQKV(QKV* m);
void saveQKV(const QKV* m, const std::string& dir);