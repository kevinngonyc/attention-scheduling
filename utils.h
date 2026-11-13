#include <string>

void printMatrix(const float* data, int stride, int rows, int cols);
void saveData(const std::string& dir, const std::string& name, const void* data, size_t bytes);
void check(cudaError_t err);