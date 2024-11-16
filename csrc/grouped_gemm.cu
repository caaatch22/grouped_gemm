#include "grouped_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

namespace grouped_gemm {

#define NUM_STREAM 4

#define CUDA_CALL(code)                                                        \
  do {                                                                         \
    cudaError_t status = code;                                                 \
    std::string err = cudaGetErrorString(status);                              \
    TORCH_CHECK(status == cudaSuccess, err);                                   \
  } while (0)

#define CUBLAS_CALL(code)                                                      \
  do {                                                                         \
    cublasStatus_t status = code;                                              \
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CuBLAS Error");              \
  } while (0)

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                                                 \
  if (error != hipSuccess) {                                                   \
    fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n",                          \
            hipGetErrorString(error), error, __FILE__, __LINE__);              \
    exit(EXIT_FAILURE);                                                        \
  }
#endif
#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                           \
  if (error != HIPBLAS_STATUS_SUCCESS) {                                       \
    fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__,     \
            __LINE__);                                                         \
    fprintf(stderr, "\n");                                                     \
    exit(EXIT_FAILURE);                                                        \
  }
#endif

#define GROUPED_GEMM_STRINGIFY_HELPER(x) #x
#define GROUPED_GEMM_STRINGIFY(x) GROUPED_GEMM_STRINGIFY_HELPER(x)

template <typename T>
torch::Tensor CopyToDevice(const std::vector<T> &x,
                           const torch::Device &device) {
  size_t bytes = x.size() * sizeof(T);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
  torch::Tensor out = torch::empty(bytes, options);

  CUDA_CALL(cudaMemcpyAsync(out.data_ptr(), x.data(), bytes,
                            cudaMemcpyHostToDevice,
                            c10::cuda::getCurrentCUDAStream()));
  return out;
}

hipblasLtHandle_t cublas_handle[NUM_STREAM];
cudaStream_t cublas_stream[NUM_STREAM];
cudaEvent_t cublas_event[NUM_STREAM];
bool cublas_init = false;

void cublas_handle_init() {
  cublas_init = true;

  for (int i = 0; i < NUM_STREAM; i++) {
    cudaStreamCreateWithFlags(&cublas_stream[i], cudaStreamNonBlocking);
    hipblasLtCreate(&cublas_handle[i]);
    cudaEventCreate(&cublas_event[i]);
  }
}

inline void cublas_current_wait_streams(cudaStream_t stream) {
  for (int s = 0; s < NUM_STREAM; s++) {
    cudaEventRecord(cublas_event[s], cublas_stream[s]);
  }

  for (int s = 0; s < NUM_STREAM; s++) {
    cudaStreamWaitEvent(stream, cublas_event[s]);
  }
}

inline void cublas_streams_wait_current(cudaStream_t stream) {
  cudaEventRecord(cublas_event[0], stream);

  for (int s = 0; s < NUM_STREAM; s++) {
    cudaStreamWaitEvent(cublas_stream[s], cublas_event[0]);
  }
}

void gemmex_wrapper_bf16(hipblasHandle_t handle, hipblasOperation_t transa,
                         hipblasOperation_t transb, int m, int n, int k,
                         int batch_count, float &alpha, float &beta,
                         c10::BFloat16 *A, c10::BFloat16 *B, c10::BFloat16 *C,
                         c10::BFloat16 *D, void *d_workspace,
                         int64_t max_workspace_size, hipStream_t stream) {
  hipblasLtMatrixLayout_t matA, matB, matC, matD;
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutCreate(&matA, HIP_R_16BF, m, k, m));
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutCreate(&matB, HIP_R_16BF, n, k, n));
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutCreate(&matC, HIP_R_16BF, m, n, m));
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutCreate(&matD, HIP_R_16BF, m, n, m));

  if (batch_count > 1) {
    int64_t stride_a = m * k;
    int64_t stride_b = k * n;
    int64_t stride_c = m * n;
    int64_t stride_d = m * n;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a,
        sizeof(stride_a)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b,
        sizeof(stride_b)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c,
        sizeof(stride_c)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d,
        sizeof(stride_d)));
  }

  hipblasLtMatmulDesc_t matmul;
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(int32_t)));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(int32_t)));

  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  // Set User Preference attributes
  hipblasLtMatmulPreference_t pref;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size,
      sizeof(max_workspace_size)));

  const int request_solutions = 1;
  hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
  int returnedAlgoCount = 0;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(
      handle, matmul, matA, matB, matC, matD, pref, request_solutions,
      heuristicResult, &returnedAlgoCount));

  if (returnedAlgoCount == 0) {
    std::cerr << "No valid solution found!" << std::endl;
    return;
  }

  uint64_t workspace_size = 0;
  for (int i = 0; i < returnedAlgoCount; i++)
    workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
  // In this sample, the workspace is already allocated with max_workspace_size
  // If not, allocate d_workspace here
  // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(
      handle, matmul, &alpha, A, matA, B, matB, &beta, C, matC, D, matD,
      &heuristicResult[0].algo, d_workspace, workspace_size, stream));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
  return;
}

// void gemmex_wrapper_bf16(hipblasHandle_t handle, hipblasOperation_t transa,
//                          hipblasOperation_t transb, int m, int n, int k,
//                          int batch_count, float &alpha, float &beta,
//                          c10::BFloat16 *A, c10::BFloat16 *B, c10::BFloat16
//                          *C, c10::BFloat16 *D, void *d_workspace, int64_t
//                          max_workspace_size, hipStream_t stream);

void CublasGemm(cublasHandle_t cublas_handle, c10::BFloat16 *a, int64_t a_rows,
                int64_t a_cols, bool trans_a, c10::BFloat16 *b, int64_t b_rows,
                int64_t b_cols, bool trans_b, c10::BFloat16 *c, int64_t c_rows,
                int64_t c_cols, cudaStream_t cublas_stream) {
  int m = trans_b ? b_rows : b_cols;
  int k = trans_b ? b_cols : b_rows;
  int n = trans_a ? a_cols : a_rows;

  int lda = trans_a ? n : k;
  int ldb = trans_b ? k : m;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  float alpha = 1.0, beta = 0.0;
  const int batch_count = 0;
  int64_t max_workspace_size = 32 * 1024 * 1024;
  void *d_workspace;
  if (max_workspace_size > 0)
    CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));

  gemmex_wrapper_bf16(cublas_handle, transpose_a, transpose_b, m, n, k,
                      batch_count, alpha, beta, a, b, c, c, d_workspace,
                      max_workspace_size, cublas_stream);

  // CUBLAS_CALL(cublasGemmEx(cublas_handle, transpose_b, transpose_a, m, n, k,
  //                          &alpha, b, CUDA_R_16BF, ldb, a, CUDA_R_16BF, lda,
  //                          &beta, c, CUDA_R_16BF, c_cols,
  //                          HIPBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void CublasGroupedGemm(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                       torch::Tensor batch_sizes, bool trans_b) {
  if (!cublas_init)
    cublas_handle_init();

  int64_t bs = batch_sizes.size(0), k = a.size(1);
  int64_t n = trans_b ? b.size(1) : b.size(2);
  int64_t b_rows = b.size(1), b_cols = b.size(2);
  c10::BFloat16 *a_ptr = a.data_ptr<c10::BFloat16>();
  c10::BFloat16 *b_ptr = b.data_ptr<c10::BFloat16>();
  c10::BFloat16 *c_ptr = c.data_ptr<c10::BFloat16>();

  cublas_streams_wait_current(c10::cuda::getCurrentCUDAStream());

  for (int i = 0; i < bs; ++i) {

    int64_t m = batch_sizes.data_ptr<int64_t>()[i];
    CublasGemm(cublas_handle[i % NUM_STREAM], a_ptr, m, k, /*trans_a=*/false,
               b_ptr, b_rows, b_cols, trans_b, c_ptr, m, n,
               cublas_stream[i % NUM_STREAM]);
    a_ptr += m * k;
    b_ptr += b_rows * b_cols;
    c_ptr += m * n;
  }

  cublas_current_wait_streams(c10::cuda::getCurrentCUDAStream());
}

void CublasGroupedGemmVariableK(torch::Tensor a, torch::Tensor b,
                                torch::Tensor c, torch::Tensor batch_sizes) {
  if (!cublas_init)
    cublas_handle_init();

  int64_t bs = batch_sizes.size(0), m = a.size(1), n = b.size(1);
  c10::BFloat16 *a_ptr = a.data_ptr<c10::BFloat16>();
  c10::BFloat16 *b_ptr = b.data_ptr<c10::BFloat16>();
  c10::BFloat16 *c_ptr = c.data_ptr<c10::BFloat16>();

  cublas_streams_wait_current(c10::cuda::getCurrentCUDAStream());

  for (int i = 0; i < bs; ++i) {
    int64_t k = batch_sizes.data_ptr<int64_t>()[i];
    CublasGemm(cublas_handle[i % NUM_STREAM], a_ptr, k, m, /*trans_a=*/true,
               b_ptr, k, n, /*trans_b=*/false, c_ptr, m, n,
               cublas_stream[i % NUM_STREAM]);
    a_ptr += k * m;
    b_ptr += k * n;
    c_ptr += m * n;
  }

  cublas_current_wait_streams(c10::cuda::getCurrentCUDAStream());
}

void GroupedGemmVariableK(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                          torch::Tensor batch_sizes) {
  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_out) for 'b'.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 2);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);

  // Validate the dimensions.
  int64_t tokens = a.size(0), num_experts = batch_sizes.size(0);
  int64_t m = a.size(1), n = b.size(1);

  // Validate that we have the same contraction dimension.
  TORCH_CHECK(tokens == b.size(0));

  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 3);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.size(0) == num_experts);
  TORCH_CHECK(c.size(1) == m);
  TORCH_CHECK(c.size(2) == n);

  // Run the computation.
  CublasGroupedGemmVariableK(a, b, c, batch_sizes);
}

// NOTE: We only support dynamic group sizes for the 'a' tensor. Tensor 'b' is
// assumed to be batched with fixed sized batches.
//
// TODO(tgale): Validate alignment is true for every batch element.
void GroupedGemm(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                 torch::Tensor batch_sizes, bool trans_a, bool trans_b) {
  // NOTE: We only support 'trans_a' or 'trans_b', not both.
  TORCH_CHECK(!(trans_a && trans_b));

  // We expect the batch_sizes on CPU.
  TORCH_CHECK(batch_sizes.is_cpu());
  TORCH_CHECK(batch_sizes.ndimension() == 1);
  TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64);

  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_in) for 'a'.
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.ndimension() == 2);
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16);

  // Defer to the variable 'k' helper for the rest of the op.
  if (trans_a) {
    GroupedGemmVariableK(a, b, c, batch_sizes);
    return;
  }

  // We expected a CUDA tensor with three dimensions and shape
  // (num_experts, hidden_in, hidden_out) for 'b'.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 3);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);

  // Validate the contraction dimensions match.
  int64_t tokens = a.size(0), num_experts = b.size(0);
  int64_t hidden_in = trans_b ? b.size(2) : b.size(1);
  int64_t hidden_out = trans_b ? b.size(1) : b.size(2);
  TORCH_CHECK(hidden_in == a.size(1));

  // Validate that we have one size per expert.
  TORCH_CHECK(batch_sizes.size(0) == num_experts);

  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 2);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.size(0) == tokens);
  TORCH_CHECK(c.size(1) == hidden_out);

  // NOTE: We support transposition through the 'trans_b' flag.
  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(b.is_contiguous());

  // NOTE: Use cuBLAS for SM90 until CUTLASS supports SM90-optimized
  // grouped-gemm.
  CublasGroupedGemm(a, b, c, batch_sizes, trans_b);
}

} // namespace grouped_gemm