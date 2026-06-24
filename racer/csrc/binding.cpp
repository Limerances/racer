#include <torch/extension.h>
#include <vector>

torch::Tensor gf256_matmul_cuda(torch::Tensor data, torch::Tensor matrix);
torch::Tensor apply_matrix_cuda_table(torch::Tensor data, torch::Tensor matrix, torch::Tensor mul_table);
torch::Tensor gf256_mul_cuda(torch::Tensor src, int64_t coeff);
torch::Tensor gf256_mul_xor_cuda(torch::Tensor src, torch::Tensor dst, int64_t coeff);
torch::Tensor xor_inplace_cuda(torch::Tensor dst, torch::Tensor src);
std::vector<torch::Tensor> apply_matrix_cuda(
    std::vector<torch::Tensor> inputs,
    torch::Tensor matrix,
    std::vector<torch::Tensor> outputs);
std::vector<torch::Tensor> apply_matrix_cuda_vector_table(
    std::vector<torch::Tensor> inputs,
    torch::Tensor matrix,
    std::vector<torch::Tensor> outputs,
    torch::Tensor mul_table);
std::vector<torch::Tensor> apply_bitmatrix_cuda(
    std::vector<torch::Tensor> inputs,
    torch::Tensor bitmatrix,
    std::vector<torch::Tensor> outputs);

torch::Tensor gf256_matmul(torch::Tensor data, torch::Tensor matrix) {
  TORCH_CHECK(data.is_cuda(), "data must be a CUDA tensor");
  TORCH_CHECK(matrix.is_cuda(), "matrix must be a CUDA tensor");
  return gf256_matmul_cuda(data, matrix);
}

torch::Tensor gf256_mul(torch::Tensor src, int64_t coeff) {
  TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
  return gf256_mul_cuda(src, coeff);
}

torch::Tensor gf256_mul_xor(torch::Tensor src, torch::Tensor dst, int64_t coeff) {
  TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
  TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
  return gf256_mul_xor_cuda(src, dst, coeff);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gf256_matmul", &gf256_matmul, "GF(2^8) matrix multiply over byte rows (CUDA)");
  m.def("apply_matrix_cuda_table", &apply_matrix_cuda_table, "Apply a GF(2^8) matrix using a device multiplication table");
  m.def("gf256_mul", &gf256_mul, "GF(2^8) multiply a uint8 CUDA tensor by a coefficient");
  m.def("gf256_mul_xor", &gf256_mul_xor, "dst ^= coeff * src over GF(2^8)");
  m.def("xor_inplace", &xor_inplace_cuda, "dst ^= src for uint8 CUDA tensors");
  m.def("apply_matrix_cuda", &apply_matrix_cuda, "Apply a GF(2^8) coefficient matrix to CUDA byte buffers");
  m.def("apply_matrix_cuda_vector_table", &apply_matrix_cuda_vector_table,
        "Apply a GF(2^8) coefficient matrix to CUDA byte buffers using a device multiplication table");
  m.def("apply_bitmatrix_cuda", &apply_bitmatrix_cuda,
        "Apply an expanded GF(2) bitmatrix to CUDA byte buffers using XOR logic");
}
