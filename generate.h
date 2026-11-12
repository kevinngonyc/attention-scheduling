#include <string>

struct QKV;

QKV* createQKV(const int* cu_seqlens, int num_seqs);
void printQKV(const QKV* m, int rows, int cols);
void freeQKV(QKV* m);
void saveQKV(const QKV* m, const std::string& dir);