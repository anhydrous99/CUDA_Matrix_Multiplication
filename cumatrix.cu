#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>
#include <stdexcept>

#include "cumatrix.h"

std::ostream& operator << (std::ostream& out, const cumatrix& mat) {
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      out << " " << mat(i, j);
    }
    out << std::endl;
  }
  return out;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) throw new std::runtime_error("");
  }
}

void cumatrix::fill_rand() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis((float)-1.0, 1.0);
  for (int i = 0; i < size(); i++)
    (*this)[i] = dis(gen);
}

float* cumatrix::get_device_pointer(bool copy) {
  if (!in_device) {
    gpuErrchk(cudaMalloc((void **) &d_elemns, N * M * sizeof(value_type)));
    if (copy) gpuErrchk(cudaMemcpy(d_elemns, elemns, N * M * sizeof(value_type), cudaMemcpyHostToDevice));
    in_device = true;
  }
  return d_elemns;
}

void cumatrix::refresh_from_device() {
  if (in_device) gpuErrchk(cudaMemcpy(elemns, d_elemns, N * M * sizeof(value_type), cudaMemcpyDeviceToHost));
}

void cumatrix::refresh_to_device() {
  if (in_device) gpuErrchk(cudaMemcpy(d_elemns, elemns, N * M * sizeof(value_type), cudaMemcpyHostToDevice));
}

void cumatrix::release_device_data() {
  if (in_device) {
    gpuErrchk(cudaFree(d_elemns));
    in_device = false;
  }
}

cumatrix operator*(cumatrix& a, cumatrix& b) {
  cumatrix output(a.rows(), b.cols());
  const float* d_a = a.get_device_pointer();
  const float* d_b = b.get_device_pointer();
  float* d_c = output.get_device_pointer(false);
  int N = a.rows(), M = b.cols(), K = a.cols();
  int lda=N, ldb=K, ldc=N;
  const float alpha = 1;
  const float beta = 0;

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc);

  cublasDestroy(handle);
  output.refresh_from_device();
  return output;
}