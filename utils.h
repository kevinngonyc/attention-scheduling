#include <string>
#include <cuda_fp16.h>

void printMatrix(const __half* data, int stride, int rows, int cols);
void saveData(const std::string& dir, const std::string& name, const void* data, size_t bytes);