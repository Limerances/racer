#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

namespace {

__device__ __forceinline__ unsigned char gf256_mul(unsigned char a, unsigned char b) {
  unsigned char p = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (b & 1) {
      p ^= a;
    }
    const bool carry = a & 0x80;
    a <<= 1;
    if (carry) {
      a ^= 0x1d;
    }
    b >>= 1;
  }
  return p;
}

__global__ void gf256_matmul_kernel(
    const unsigned char* __restrict__ data,
    const unsigned char* __restrict__ matrix,
    unsigned char* __restrict__ out,
    int rows,
    int k,
    int64_t size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(rows) * size;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / size);
  const int64_t offset = idx - static_cast<int64_t>(row) * size;
  unsigned char acc = 0;
  for (int col = 0; col < k; ++col) {
    const unsigned char coef = matrix[row * k + col];
    if (coef != 0) {
      acc ^= gf256_mul(coef, data[static_cast<int64_t>(col) * size + offset]);
    }
  }
  out[idx] = acc;
}


__global__ void gf256_matmul_table_kernel(
    const unsigned char* __restrict__ data,
    const unsigned char* __restrict__ matrix,
    const unsigned char* __restrict__ mul_table,
    unsigned char* __restrict__ out,
    int rows,
    int k,
    int64_t size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(rows) * size;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / size);
  const int64_t offset = idx - static_cast<int64_t>(row) * size;
  unsigned char acc = 0;
  for (int col = 0; col < k; ++col) {
    const unsigned char coef = matrix[row * k + col];
    if (coef != 0) {
      const unsigned char x = data[static_cast<int64_t>(col) * size + offset];
      acc ^= mul_table[static_cast<int>(coef) * 256 + static_cast<int>(x)];
    }
  }
  out[idx] = acc;
}

__global__ void gf256_mul_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned char coeff,
    int64_t size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  dst[idx] = gf256_mul(coeff, src[idx]);
}

__global__ void gf256_mul_xor_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned char coeff,
    int64_t size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  dst[idx] ^= gf256_mul(coeff, src[idx]);
}

__global__ void xor_inplace_kernel(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    int64_t size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  dst[idx] ^= src[idx];
}

__global__ void apply_matrix_vector_kernel(
    const unsigned char* const* __restrict__ inputs,
    const unsigned char* __restrict__ matrix,
    unsigned char** __restrict__ outputs,
    int rows,
    int k,
    int64_t size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(rows) * size;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / size);
  const int64_t offset = idx - static_cast<int64_t>(row) * size;
  unsigned char acc = 0;
  for (int col = 0; col < k; ++col) {
    const unsigned char c = matrix[row * k + col];
    if (c != 0) {
      acc ^= gf256_mul(c, inputs[col][offset]);
    }
  }
  outputs[row][offset] = acc;
}


__global__ void apply_matrix_vector_table_kernel(
    const unsigned char* const* __restrict__ inputs,
    const unsigned char* __restrict__ matrix,
    const unsigned char* __restrict__ mul_table,
    unsigned char** __restrict__ outputs,
    int rows,
    int k,
    int64_t size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(rows) * size;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / size);
  const int64_t offset = idx - static_cast<int64_t>(row) * size;
  unsigned char acc = 0;
  for (int col = 0; col < k; ++col) {
    const unsigned char c = matrix[row * k + col];
    if (c != 0) {
      const unsigned char x = inputs[col][offset];
      acc ^= mul_table[static_cast<int>(c) * 256 + static_cast<int>(x)];
    }
  }
  outputs[row][offset] = acc;
}

__global__ void apply_bitmatrix_vector_kernel(
    const unsigned char* const* __restrict__ inputs,
    const unsigned char* __restrict__ bitmatrix,
    unsigned char** __restrict__ outputs,
    int rows,
    int k,
    int64_t size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(rows) * size;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / size);
  const int64_t offset = idx - static_cast<int64_t>(row) * size;
  unsigned char out = 0;
  for (int out_bit = 0; out_bit < 8; ++out_bit) {
    unsigned char bit = 0;
    const int bm_row = row * 8 + out_bit;
    for (int col = 0; col < k; ++col) {
      const unsigned char value = inputs[col][offset];
      const int bm_col_base = col * 8;
      #pragma unroll
      for (int in_bit = 0; in_bit < 8; ++in_bit) {
        const unsigned char enabled = bitmatrix[bm_row * k * 8 + bm_col_base + in_bit];
        bit ^= static_cast<unsigned char>(enabled & ((value >> in_bit) & 1));
      }
    }
    out |= static_cast<unsigned char>(bit << out_bit);
  }
  outputs[row][offset] = out;
}

__global__ void apply_matrix_k3_table_kernel(
    const unsigned char* __restrict__ in0,
    const unsigned char* __restrict__ in1,
    const unsigned char* __restrict__ in2,
    const unsigned char* __restrict__ matrix,
    const unsigned char* __restrict__ mul_table,
    unsigned char* __restrict__ out0,
    unsigned char* __restrict__ out1,
    unsigned char* __restrict__ out2,
    int rows,
    int64_t size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(rows) * size;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / size);
  const int64_t offset = idx - static_cast<int64_t>(row) * size;
  const unsigned char c0 = matrix[row * 3];
  const unsigned char c1 = matrix[row * 3 + 1];
  const unsigned char c2 = matrix[row * 3 + 2];
  unsigned char acc = 0;
  if (c0 != 0) {
    const unsigned char x = in0[offset];
    acc ^= mul_table[static_cast<int>(c0) * 256 + static_cast<int>(x)];
  }
  if (c1 != 0) {
    const unsigned char x = in1[offset];
    acc ^= mul_table[static_cast<int>(c1) * 256 + static_cast<int>(x)];
  }
  if (c2 != 0) {
    const unsigned char x = in2[offset];
    acc ^= mul_table[static_cast<int>(c2) * 256 + static_cast<int>(x)];
  }
  unsigned char* out = row == 0 ? out0 : (row == 1 ? out1 : out2);
  out[offset] = acc;
}

void check_uint8_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8, name, " must be torch.uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}


void check_mul_table(const torch::Tensor& tensor, const c10::Device& device) {
  check_uint8_cuda_contiguous(tensor, "mul_table");
  TORCH_CHECK(tensor.dim() == 2, "mul_table must have shape [256, 256]");
  TORCH_CHECK(tensor.size(0) == 256 && tensor.size(1) == 256,
              "mul_table must have shape [256, 256]");
  TORCH_CHECK(tensor.device() == device, "mul_table must be on the same CUDA device as inputs");
}

int launch_blocks(int64_t size, int threads) {
  return static_cast<int>((size + threads - 1) / threads);
}

}  // namespace

torch::Tensor gf256_matmul_cuda(torch::Tensor data, torch::Tensor matrix) {
  TORCH_CHECK(data.is_cuda(), "data must be CUDA");
  TORCH_CHECK(matrix.is_cuda(), "matrix must be CUDA");
  TORCH_CHECK(data.scalar_type() == torch::kUInt8, "data must be torch.uint8");
  TORCH_CHECK(matrix.scalar_type() == torch::kUInt8, "matrix must be torch.uint8");
  TORCH_CHECK(data.dim() == 2, "data must have shape [k, size]");
  TORCH_CHECK(matrix.dim() == 2, "matrix must have shape [rows, k]");
  TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
  TORCH_CHECK(matrix.is_contiguous(), "matrix must be contiguous");
  TORCH_CHECK(data.size(0) == matrix.size(1), "matrix width must equal data rows");

  const int k = static_cast<int>(data.size(0));
  const int rows = static_cast<int>(matrix.size(0));
  const int64_t size = data.size(1);
  auto out = torch::empty({rows, size}, data.options());
  if (rows == 0 || size == 0) {
    return out;
  }

  const int threads = 256;
  const int64_t total = static_cast<int64_t>(rows) * size;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gf256_matmul_kernel<<<blocks, threads, 0, stream>>>(
      data.data_ptr<unsigned char>(),
      matrix.data_ptr<unsigned char>(),
      out.data_ptr<unsigned char>(),
      rows,
      k,
      size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


torch::Tensor apply_matrix_cuda_table(torch::Tensor data, torch::Tensor matrix, torch::Tensor mul_table) {
  TORCH_CHECK(data.is_cuda(), "data must be CUDA");
  TORCH_CHECK(matrix.is_cuda(), "matrix must be CUDA");
  TORCH_CHECK(mul_table.is_cuda(), "mul_table must be CUDA");
  TORCH_CHECK(data.scalar_type() == torch::kUInt8, "data must be torch.uint8");
  TORCH_CHECK(matrix.scalar_type() == torch::kUInt8, "matrix must be torch.uint8");
  TORCH_CHECK(mul_table.scalar_type() == torch::kUInt8, "mul_table must be torch.uint8");
  TORCH_CHECK(data.dim() == 2, "data must have shape [k, size]");
  TORCH_CHECK(matrix.dim() == 2, "matrix must have shape [rows, k]");
  TORCH_CHECK(mul_table.dim() == 2, "mul_table must have shape [256, 256]");
  TORCH_CHECK(mul_table.size(0) == 256 && mul_table.size(1) == 256,
              "mul_table must have shape [256, 256]");
  TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
  TORCH_CHECK(matrix.is_contiguous(), "matrix must be contiguous");
  TORCH_CHECK(mul_table.is_contiguous(), "mul_table must be contiguous");
  TORCH_CHECK(data.size(0) == matrix.size(1), "matrix width must equal data rows");
  TORCH_CHECK(data.device() == matrix.device(), "matrix must be on the same CUDA device as data");
  TORCH_CHECK(data.device() == mul_table.device(), "mul_table must be on the same CUDA device as data");

  c10::cuda::CUDAGuard guard(data.device());
  const int k = static_cast<int>(data.size(0));
  const int rows = static_cast<int>(matrix.size(0));
  const int64_t size = data.size(1);
  auto out = torch::empty({rows, size}, data.options());
  if (rows == 0 || size == 0) {
    return out;
  }

  const int threads = 256;
  const int64_t total = static_cast<int64_t>(rows) * size;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gf256_matmul_table_kernel<<<blocks, threads, 0, stream>>>(
      data.data_ptr<unsigned char>(),
      matrix.data_ptr<unsigned char>(),
      mul_table.data_ptr<unsigned char>(),
      out.data_ptr<unsigned char>(),
      rows,
      k,
      size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor gf256_mul_cuda(torch::Tensor src, int64_t coeff) {
  check_uint8_cuda_contiguous(src, "src");
  auto dst = torch::empty_like(src);
  const int64_t size = src.numel();
  if (size == 0) {
    return dst;
  }
  c10::cuda::CUDAGuard guard(src.device());
  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  gf256_mul_kernel<<<launch_blocks(size, threads), threads, 0, stream>>>(
      src.data_ptr<unsigned char>(),
      dst.data_ptr<unsigned char>(),
      static_cast<unsigned char>(coeff & 0xff),
      size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dst;
}

torch::Tensor gf256_mul_xor_cuda(torch::Tensor src, torch::Tensor dst, int64_t coeff) {
  check_uint8_cuda_contiguous(src, "src");
  check_uint8_cuda_contiguous(dst, "dst");
  TORCH_CHECK(src.device() == dst.device(), "src and dst must be on the same CUDA device");
  TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same numel");
  const int64_t size = src.numel();
  if (size == 0) {
    return dst;
  }
  c10::cuda::CUDAGuard guard(src.device());
  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  gf256_mul_xor_kernel<<<launch_blocks(size, threads), threads, 0, stream>>>(
      src.data_ptr<unsigned char>(),
      dst.data_ptr<unsigned char>(),
      static_cast<unsigned char>(coeff & 0xff),
      size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dst;
}

torch::Tensor xor_inplace_cuda(torch::Tensor dst, torch::Tensor src) {
  check_uint8_cuda_contiguous(dst, "dst");
  check_uint8_cuda_contiguous(src, "src");
  TORCH_CHECK(src.device() == dst.device(), "src and dst must be on the same CUDA device");
  TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same numel");
  const int64_t size = src.numel();
  if (size == 0) {
    return dst;
  }
  c10::cuda::CUDAGuard guard(dst.device());
  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  xor_inplace_kernel<<<launch_blocks(size, threads), threads, 0, stream>>>(
      dst.data_ptr<unsigned char>(),
      src.data_ptr<unsigned char>(),
      size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dst;
}

std::vector<torch::Tensor> apply_matrix_cuda(
    std::vector<torch::Tensor> inputs,
    torch::Tensor matrix,
    std::vector<torch::Tensor> outputs) {
  TORCH_CHECK(!inputs.empty(), "inputs must be non-empty");
  TORCH_CHECK(!outputs.empty(), "outputs must be non-empty");
  check_uint8_cuda_contiguous(matrix, "matrix");
  TORCH_CHECK(matrix.dim() == 2, "matrix must have shape [outputs, inputs]");
  TORCH_CHECK(matrix.size(0) == static_cast<int64_t>(outputs.size()), "matrix row count must match outputs");
  TORCH_CHECK(matrix.size(1) == static_cast<int64_t>(inputs.size()), "matrix width must match inputs");

  const auto device = inputs[0].device();
  const int64_t size = inputs[0].numel();
  TORCH_CHECK(matrix.device() == device, "matrix must be on the same CUDA device as inputs");
  for (size_t i = 0; i < inputs.size(); ++i) {
    check_uint8_cuda_contiguous(inputs[i], "input");
    TORCH_CHECK(inputs[i].device() == device, "all inputs must be on the same CUDA device");
    TORCH_CHECK(inputs[i].numel() == size, "all inputs must have the same numel");
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    check_uint8_cuda_contiguous(outputs[i], "output");
    TORCH_CHECK(outputs[i].device() == device, "all outputs must be on the same CUDA device as inputs");
    TORCH_CHECK(outputs[i].numel() == size, "all outputs must have the same numel as inputs");
  }

  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = static_cast<int64_t>(outputs.size()) * size;
  const int blocks = launch_blocks(total, threads);

  std::vector<const unsigned char*> h_inputs;
  std::vector<unsigned char*> h_outputs;
  h_inputs.reserve(inputs.size());
  h_outputs.reserve(outputs.size());
  for (const auto& input : inputs) {
    h_inputs.push_back(input.data_ptr<unsigned char>());
  }
  for (auto& output : outputs) {
    h_outputs.push_back(output.data_ptr<unsigned char>());
  }

  const unsigned char** d_inputs = nullptr;
  unsigned char** d_outputs = nullptr;
  cudaMalloc(&d_inputs, sizeof(unsigned char*) * h_inputs.size());
  cudaMalloc(&d_outputs, sizeof(unsigned char*) * h_outputs.size());
  cudaMemcpyAsync(d_inputs, h_inputs.data(), sizeof(unsigned char*) * h_inputs.size(), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_outputs, h_outputs.data(), sizeof(unsigned char*) * h_outputs.size(), cudaMemcpyHostToDevice, stream);

  apply_matrix_vector_kernel<<<blocks, threads, 0, stream>>>(
      d_inputs,
      matrix.data_ptr<unsigned char>(),
      d_outputs,
      static_cast<int>(outputs.size()),
      static_cast<int>(inputs.size()),
      size);
  cudaFree(d_inputs);
  cudaFree(d_outputs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}


std::vector<torch::Tensor> apply_matrix_cuda_vector_table(
    std::vector<torch::Tensor> inputs,
    torch::Tensor matrix,
    std::vector<torch::Tensor> outputs,
    torch::Tensor mul_table) {
  TORCH_CHECK(!inputs.empty(), "inputs must be non-empty");
  TORCH_CHECK(!outputs.empty(), "outputs must be non-empty");
  check_uint8_cuda_contiguous(matrix, "matrix");
  TORCH_CHECK(matrix.dim() == 2, "matrix must have shape [outputs, inputs]");
  TORCH_CHECK(matrix.size(0) == static_cast<int64_t>(outputs.size()), "matrix row count must match outputs");
  TORCH_CHECK(matrix.size(1) == static_cast<int64_t>(inputs.size()), "matrix width must match inputs");

  const auto device = inputs[0].device();
  const int64_t size = inputs[0].numel();
  TORCH_CHECK(matrix.device() == device, "matrix must be on the same CUDA device as inputs");
  check_mul_table(mul_table, device);
  for (const auto& input : inputs) {
    check_uint8_cuda_contiguous(input, "input");
    TORCH_CHECK(input.device() == device, "all inputs must be on the same CUDA device");
    TORCH_CHECK(input.numel() == size, "all inputs must have the same numel");
  }
  for (const auto& output : outputs) {
    check_uint8_cuda_contiguous(output, "output");
    TORCH_CHECK(output.device() == device, "all outputs must be on the same CUDA device as inputs");
    TORCH_CHECK(output.numel() == size, "all outputs must have the same numel as inputs");
  }

  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = static_cast<int64_t>(outputs.size()) * size;
  const int blocks = launch_blocks(total, threads);
  if (total == 0) {
    return outputs;
  }

  if (inputs.size() == 3 && outputs.size() <= 3) {
    unsigned char* out0 = outputs[0].data_ptr<unsigned char>();
    unsigned char* out1 = outputs.size() > 1 ? outputs[1].data_ptr<unsigned char>() : out0;
    unsigned char* out2 = outputs.size() > 2 ? outputs[2].data_ptr<unsigned char>() : out0;
    apply_matrix_k3_table_kernel<<<blocks, threads, 0, stream>>>(
        inputs[0].data_ptr<unsigned char>(),
        inputs[1].data_ptr<unsigned char>(),
        inputs[2].data_ptr<unsigned char>(),
        matrix.data_ptr<unsigned char>(),
        mul_table.data_ptr<unsigned char>(),
        out0,
        out1,
        out2,
        static_cast<int>(outputs.size()),
        size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return outputs;
  }

  std::vector<const unsigned char*> h_inputs;
  std::vector<unsigned char*> h_outputs;
  h_inputs.reserve(inputs.size());
  h_outputs.reserve(outputs.size());
  for (const auto& input : inputs) {
    h_inputs.push_back(input.data_ptr<unsigned char>());
  }
  for (auto& output : outputs) {
    h_outputs.push_back(output.data_ptr<unsigned char>());
  }

  const unsigned char** d_inputs = nullptr;
  unsigned char** d_outputs = nullptr;
  cudaMalloc(&d_inputs, sizeof(unsigned char*) * h_inputs.size());
  cudaMalloc(&d_outputs, sizeof(unsigned char*) * h_outputs.size());
  cudaMemcpyAsync(d_inputs, h_inputs.data(), sizeof(unsigned char*) * h_inputs.size(), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_outputs, h_outputs.data(), sizeof(unsigned char*) * h_outputs.size(), cudaMemcpyHostToDevice, stream);

  apply_matrix_vector_table_kernel<<<blocks, threads, 0, stream>>>(
      d_inputs,
      matrix.data_ptr<unsigned char>(),
      mul_table.data_ptr<unsigned char>(),
      d_outputs,
      static_cast<int>(outputs.size()),
      static_cast<int>(inputs.size()),
      size);
  cudaFree(d_inputs);
  cudaFree(d_outputs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

std::vector<torch::Tensor> apply_bitmatrix_cuda(
    std::vector<torch::Tensor> inputs,
    torch::Tensor bitmatrix,
    std::vector<torch::Tensor> outputs) {
  TORCH_CHECK(!inputs.empty(), "inputs must be non-empty");
  TORCH_CHECK(!outputs.empty(), "outputs must be non-empty");
  check_uint8_cuda_contiguous(bitmatrix, "bitmatrix");
  TORCH_CHECK(bitmatrix.dim() == 2, "bitmatrix must have shape [outputs * 8, inputs * 8]");
  TORCH_CHECK(bitmatrix.size(0) == static_cast<int64_t>(outputs.size()) * 8,
              "bitmatrix row count must equal outputs * 8");
  TORCH_CHECK(bitmatrix.size(1) == static_cast<int64_t>(inputs.size()) * 8,
              "bitmatrix width must equal inputs * 8");

  const auto device = inputs[0].device();
  const int64_t size = inputs[0].numel();
  TORCH_CHECK(bitmatrix.device() == device, "bitmatrix must be on the same CUDA device as inputs");
  for (const auto& input : inputs) {
    check_uint8_cuda_contiguous(input, "input");
    TORCH_CHECK(input.device() == device, "all inputs must be on the same CUDA device");
    TORCH_CHECK(input.numel() == size, "all inputs must have the same numel");
  }
  for (const auto& output : outputs) {
    check_uint8_cuda_contiguous(output, "output");
    TORCH_CHECK(output.device() == device, "all outputs must be on the same CUDA device as inputs");
    TORCH_CHECK(output.numel() == size, "all outputs must have the same numel as inputs");
  }

  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = static_cast<int64_t>(outputs.size()) * size;
  if (total == 0) {
    return outputs;
  }

  std::vector<const unsigned char*> h_inputs;
  std::vector<unsigned char*> h_outputs;
  h_inputs.reserve(inputs.size());
  h_outputs.reserve(outputs.size());
  for (const auto& input : inputs) {
    h_inputs.push_back(input.data_ptr<unsigned char>());
  }
  for (auto& output : outputs) {
    h_outputs.push_back(output.data_ptr<unsigned char>());
  }

  const unsigned char** d_inputs = nullptr;
  unsigned char** d_outputs = nullptr;
  cudaMalloc(&d_inputs, sizeof(unsigned char*) * h_inputs.size());
  cudaMalloc(&d_outputs, sizeof(unsigned char*) * h_outputs.size());
  cudaMemcpyAsync(d_inputs, h_inputs.data(), sizeof(unsigned char*) * h_inputs.size(), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_outputs, h_outputs.data(), sizeof(unsigned char*) * h_outputs.size(), cudaMemcpyHostToDevice, stream);

  apply_bitmatrix_vector_kernel<<<launch_blocks(total, threads), threads, 0, stream>>>(
      d_inputs,
      bitmatrix.data_ptr<unsigned char>(),
      d_outputs,
      static_cast<int>(outputs.size()),
      static_cast<int>(inputs.size()),
      size);
  cudaFree(d_inputs);
  cudaFree(d_outputs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}
