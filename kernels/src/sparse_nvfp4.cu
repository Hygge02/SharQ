#include "sparse_nvfp4.h"
#include "cutlass/util/device_memory.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace sparse_nvfp4 {

namespace {

inline void check_cuda(cudaError_t status, char const* expr, char const* file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line
              << " for " << expr << ": " << cudaGetErrorString(status)
              << std::endl;
    std::abort();
  }
}

#define SHARQ_CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

struct SparseShapeKey {
  int device;
  int m;
  int n;
  int k;

  bool operator==(SparseShapeKey const& other) const {
    return device == other.device && m == other.m && n == other.n && k == other.k;
  }
};

struct SparseShapeKeyHash {
  size_t operator()(SparseShapeKey const& key) const {
    size_t hash = static_cast<size_t>(key.device);
    hash = hash * 1315423911u + static_cast<size_t>(key.m);
    hash = hash * 1315423911u + static_cast<size_t>(key.n);
    hash = hash * 1315423911u + static_cast<size_t>(key.k);
    return hash;
  }
};

enum class SparseKernelVariant {
  kBaselineAuto,
#if defined(SHARQ_TARGET_SM100A)
  kSmall1Sm128x128x256,
#endif
};

template <class TileShape, class ClusterShape, class KernelSchedule, class EpilogueSchedule>
struct SparseKernelTraits {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using SparseConfig = typename CollectiveMainloop::SparseConfig;
  using LayoutA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
  using LayoutE = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

using SparseKernelBaseline = SparseKernelTraits<
    sharq::blackwell_arch::SparseTileShape,
    sharq::blackwell_arch::SparseClusterShape,
    sharq::blackwell_arch::SparseKernelSchedule,
    sharq::blackwell_arch::SparseEpilogueSchedule>;

#if defined(SHARQ_TARGET_SM100A)
using SparseKernelSmall1Sm = SparseKernelTraits<
    Shape<_128, _128, _256>,
    Shape<_1, _1, _1>,
    cutlass::gemm::KernelSparseTmaWarpSpecialized1SmNvf4Sm100,
    cutlass::epilogue::TmaWarpSpecialized1SmNvf4>;

static_assert(std::is_same_v<typename SparseKernelBaseline::SparseConfig,
                             typename SparseKernelSmall1Sm::SparseConfig>);
static_assert(std::is_same_v<typename SparseKernelBaseline::LayoutA,
                             typename SparseKernelSmall1Sm::LayoutA>);
static_assert(std::is_same_v<typename SparseKernelBaseline::LayoutE,
                             typename SparseKernelSmall1Sm::LayoutE>);
static_assert(std::is_same_v<typename SparseKernelBaseline::LayoutSFA,
                             typename SparseKernelSmall1Sm::LayoutSFA>);
static_assert(std::is_same_v<typename SparseKernelBaseline::LayoutSFB,
                             typename SparseKernelSmall1Sm::LayoutSFB>);
#endif

std::unordered_map<SparseShapeKey, SparseKernelVariant, SparseShapeKeyHash> g_sparse_variant_cache;
std::mutex g_sparse_variant_cache_mutex;

template <typename KernelTraits>
cutlass::Status launch_sparse_variant(
    const cutlass::float_e2m1_t* A,
    const cutlass::float_e2m1_t* B,
    const uint8_t* E,
    int M,
    int N,
    int K,
    cutlass::bfloat16_t* C,
    cutlass::bfloat16_t* D,
    const cutlass::float_ue4m3_t* SFA,
    const cutlass::float_ue4m3_t* SFB,
    float alpha,
    float beta,
    const float* alpha_ptr) {
  using Gemm = typename KernelTraits::Gemm;
  using SparseConfig = typename KernelTraits::SparseConfig;
  using LayoutA = typename KernelTraits::LayoutA;
  using LayoutE = typename KernelTraits::LayoutE;
  using LayoutSFA = typename KernelTraits::LayoutSFA;
  using LayoutSFB = typename KernelTraits::LayoutSFB;
  using StrideB = typename KernelTraits::StrideB;
  using StrideC = typename KernelTraits::StrideC;
  using StrideD = typename KernelTraits::StrideD;
  using Sm1xxBlkScaledConfig = typename KernelTraits::Sm1xxBlkScaledConfig;

  auto problem_shape = cute::make_shape(M, N, K, 1);
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  LayoutA layout_A = SparseConfig::fill_layoutA(problem_shape);
  LayoutE layout_E = SparseConfig::fill_layoutE(problem_shape);
  LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape);
  LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape);

  Gemm gemm_op;
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          A, layout_A,
          B, stride_B,
          E, layout_E,
          SFA, layout_SFA,
          SFB, layout_SFB,
      },
      {
          {},
          C, stride_C,
          D, stride_D,
      }};
  arguments.epilogue.thread.alpha = alpha;
  arguments.epilogue.thread.beta = beta;
  arguments.epilogue.thread.alpha_ptr = alpha_ptr;

  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  return gemm_op.run();
}

cutlass::Status dispatch_sparse_variant(
    SparseKernelVariant variant,
    const cutlass::float_e2m1_t* A,
    const cutlass::float_e2m1_t* B,
    const uint8_t* E,
    int M,
    int N,
    int K,
    cutlass::bfloat16_t* C,
    cutlass::bfloat16_t* D,
    const cutlass::float_ue4m3_t* SFA,
    const cutlass::float_ue4m3_t* SFB,
    float alpha,
    float beta,
    const float* alpha_ptr) {
  switch (variant) {
    case SparseKernelVariant::kBaselineAuto:
      return launch_sparse_variant<SparseKernelBaseline>(A, B, E, M, N, K, C, D, SFA, SFB, alpha, beta, alpha_ptr);
#if defined(SHARQ_TARGET_SM100A)
    case SparseKernelVariant::kSmall1Sm128x128x256:
      return launch_sparse_variant<SparseKernelSmall1Sm>(A, B, E, M, N, K, C, D, SFA, SFB, alpha, beta, alpha_ptr);
#endif
  }
  return cutlass::Status::kErrorInternal;
}

bool env_flag_enabled(char const* name) {
  char const* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  return value[0] != '\0' && value[0] != '0' && value[0] != 'f' && value[0] != 'F';
}

std::optional<SparseKernelVariant> forced_sparse_variant_from_env() {
  char const* value = std::getenv("SHARQ_SPARSE_NVFP4_KERNEL");
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }

  std::string_view name(value);
  if (name == "baseline_auto") {
    return SparseKernelVariant::kBaselineAuto;
  }
#if defined(SHARQ_TARGET_SM100A)
  if (name == "small_1sm_128x128x256") {
    return SparseKernelVariant::kSmall1Sm128x128x256;
  }
#endif
  return std::nullopt;
}

char const* sparse_variant_name(SparseKernelVariant variant) {
  switch (variant) {
    case SparseKernelVariant::kBaselineAuto:
      return "baseline_auto";
#if defined(SHARQ_TARGET_SM100A)
    case SparseKernelVariant::kSmall1Sm128x128x256:
      return "small_1sm_128x128x256";
#endif
  }
  return "unknown";
}

int sparse_autotune_iterations(int M, int N, int K) {
  long long work = static_cast<long long>(M) * static_cast<long long>(N) * static_cast<long long>(K);
  if (work >= 200000000000LL) {
    return 8;
  }
  if (work >= 50000000000LL) {
    return 12;
  }
  if (work >= 5000000000LL) {
    return 20;
  }
  return 40;
}

SparseKernelVariant sparse_heuristic_variant(int M, int N, int K) {
#if defined(SHARQ_TARGET_SM100A)
  (void)N;
  (void)K;
  if (M <= 256) {
    return SparseKernelVariant::kSmall1Sm128x128x256;
  }
#else
  (void)M;
  (void)N;
  (void)K;
#endif
  return SparseKernelVariant::kBaselineAuto;
}

float benchmark_sparse_variant(
    SparseKernelVariant variant,
    const cutlass::float_e2m1_t* A,
    const cutlass::float_e2m1_t* B,
    const uint8_t* E,
    int M,
    int N,
    int K,
    cutlass::bfloat16_t* C,
    cutlass::bfloat16_t* D,
    const cutlass::float_ue4m3_t* SFA,
    const cutlass::float_ue4m3_t* SFB,
    float alpha,
    float beta,
    const float* alpha_ptr) {
  constexpr int kWarmupIters = 2;
  constexpr int kTimingRepeats = 2;
  int iters = sparse_autotune_iterations(M, N, K);
  float best_ms = std::numeric_limits<float>::infinity();

  for (int repeat = 0; repeat < kTimingRepeats; ++repeat) {
    for (int i = 0; i < kWarmupIters; ++i) {
      if (dispatch_sparse_variant(variant, A, B, E, M, N, K, C, D, SFA, SFB, alpha, beta, alpha_ptr) != cutlass::Status::kSuccess) {
        return std::numeric_limits<float>::infinity();
      }
    }
    SHARQ_CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    SHARQ_CHECK_CUDA(cudaEventCreate(&start));
    SHARQ_CHECK_CUDA(cudaEventCreate(&stop));

    SHARQ_CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
      if (dispatch_sparse_variant(variant, A, B, E, M, N, K, C, D, SFA, SFB, alpha, beta, alpha_ptr) != cutlass::Status::kSuccess) {
        SHARQ_CHECK_CUDA(cudaEventDestroy(start));
        SHARQ_CHECK_CUDA(cudaEventDestroy(stop));
        return std::numeric_limits<float>::infinity();
      }
    }
    SHARQ_CHECK_CUDA(cudaEventRecord(stop));
    SHARQ_CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    SHARQ_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    SHARQ_CHECK_CUDA(cudaEventDestroy(start));
    SHARQ_CHECK_CUDA(cudaEventDestroy(stop));
    best_ms = std::min(best_ms, elapsed_ms / static_cast<float>(iters));
  }

  return best_ms;
}

SparseKernelVariant autotune_sparse_variant(
    int device,
    const cutlass::float_e2m1_t* A,
    const cutlass::float_e2m1_t* B,
    const uint8_t* E,
    int M,
    int N,
    int K,
    cutlass::bfloat16_t* C,
    cutlass::bfloat16_t* D,
    const cutlass::float_ue4m3_t* SFA,
    const cutlass::float_ue4m3_t* SFB,
    float alpha,
    float beta,
    const float* alpha_ptr) {
  SparseKernelVariant best_variant = sparse_heuristic_variant(M, N, K);
  float best_ms = std::numeric_limits<float>::infinity();

  constexpr SparseKernelVariant kCandidates[] = {
      SparseKernelVariant::kBaselineAuto,
#if defined(SHARQ_TARGET_SM100A)
      SparseKernelVariant::kSmall1Sm128x128x256,
#endif
  };

  for (SparseKernelVariant candidate : kCandidates) {
    float candidate_ms = benchmark_sparse_variant(candidate, A, B, E, M, N, K, C, D, SFA, SFB, alpha, beta, alpha_ptr);
    if (candidate_ms < best_ms) {
      best_ms = candidate_ms;
      best_variant = candidate;
    }
  }

  if (!std::isfinite(best_ms)) {
    best_variant = sparse_heuristic_variant(M, N, K);
  }

  if (env_flag_enabled("SHARQ_NVFP4_LOG_AUTOTUNE")) {
    std::cerr << "Sparse NVFP4 autotune device=" << device
              << " M=" << M << " N=" << N << " K=" << K
              << " -> " << sparse_variant_name(best_variant);
    if (std::isfinite(best_ms)) {
      std::cerr << " (" << best_ms << " ms)";
    }
    std::cerr << std::endl;
  }

  return best_variant;
}

SparseKernelVariant select_sparse_variant(
    const cutlass::float_e2m1_t* A,
    const cutlass::float_e2m1_t* B,
    const uint8_t* E,
    int M,
    int N,
    int K,
    cutlass::bfloat16_t* C,
    cutlass::bfloat16_t* D,
    const cutlass::float_ue4m3_t* SFA,
    const cutlass::float_ue4m3_t* SFB,
    float alpha,
    float beta,
    const float* alpha_ptr) {
  if (auto forced = forced_sparse_variant_from_env()) {
    return *forced;
  }

  int device = 0;
  SHARQ_CHECK_CUDA(cudaGetDevice(&device));
  SparseShapeKey key{device, M, N, K};

  {
    std::lock_guard<std::mutex> lock(g_sparse_variant_cache_mutex);
    auto it = g_sparse_variant_cache.find(key);
    if (it != g_sparse_variant_cache.end()) {
      return it->second;
    }
  }

  SparseKernelVariant selected = env_flag_enabled("SHARQ_SPARSE_NVFP4_DISABLE_AUTOTUNE")
      ? sparse_heuristic_variant(M, N, K)
      : autotune_sparse_variant(device, A, B, E, M, N, K, C, D, SFA, SFB, alpha, beta, alpha_ptr);

  {
    std::lock_guard<std::mutex> lock(g_sparse_variant_cache_mutex);
    g_sparse_variant_cache.emplace(key, selected);
  }
  return selected;
}

}  // namespace

void matmul_host_sparse_nvfp4_bf16(
    const cutlass::float_e2m1_t* A,
    const cutlass::float_e2m1_t* B,
    const uint8_t* E,
    int M,
    int N,
    int K,
    cutlass::bfloat16_t* C,
    cutlass::bfloat16_t* D,
    const cutlass::float_ue4m3_t* SFA,
    const cutlass::float_ue4m3_t* SFB,
    float alpha,
    float beta,
    const float* alpha_ptr) {
  SparseKernelVariant variant = select_sparse_variant(A, B, E, M, N, K, C, D, SFA, SFB, alpha, beta, alpha_ptr);
  auto status = dispatch_sparse_variant(variant, A, B, E, M, N, K, C, D, SFA, SFB, alpha, beta, alpha_ptr);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS sparse NVFP4 GEMM failed with status: "
              << cutlass::cutlassGetStatusString(status) << " ("
              << static_cast<int>(status) << ")"
              << ", variant=" << sparse_variant_name(variant)
              << std::endl;
  }
  assert(status == cutlass::Status::kSuccess);
}

}  // namespace sparse_nvfp4
