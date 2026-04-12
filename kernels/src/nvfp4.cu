#include "nvfp4.h"
#include "sharq_blackwell_arch.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>

using namespace cute;

namespace {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag = sharq::blackwell_arch::DenseArchTag;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

struct DenseShapeKey {
  int device;
  int m;
  int n;
  int k;

  bool operator==(DenseShapeKey const& other) const {
    return device == other.device && m == other.m && n == other.n && k == other.k;
  }
};

struct DenseShapeKeyHash {
  size_t operator()(DenseShapeKey const& key) const {
    size_t hash = static_cast<size_t>(key.device);
    hash = hash * 1315423911u + static_cast<size_t>(key.m);
    hash = hash * 1315423911u + static_cast<size_t>(key.n);
    hash = hash * 1315423911u + static_cast<size_t>(key.k);
    return hash;
  }
};

enum class DenseKernelVariant {
  kBaselineAuto,
#if defined(SHARQ_TARGET_SM100A)
  kSmall1Sm128x128x256,
#endif
};

template <class TileShape, class ClusterShape, class KernelSchedule, class EpilogueSchedule>
struct DenseKernelTraits {
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
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

using DenseKernelBaseline = DenseKernelTraits<
    sharq::blackwell_arch::DenseTileShape,
    sharq::blackwell_arch::DenseClusterShape,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto>;

#if defined(SHARQ_TARGET_SM100A)
using DenseKernelSmall1Sm = DenseKernelTraits<
    Shape<_128, _128, _256>,
    Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100,
    cutlass::epilogue::TmaWarpSpecialized1SmNvf4>;

static_assert(std::is_same_v<typename DenseKernelBaseline::Sm1xxBlkScaledConfig,
                             typename DenseKernelSmall1Sm::Sm1xxBlkScaledConfig>);
#endif

std::unordered_map<DenseShapeKey, DenseKernelVariant, DenseShapeKeyHash> g_dense_variant_cache;
std::mutex g_dense_variant_cache_mutex;

template <typename KernelTraits>
cutlass::Status launch_dense_variant(
    const ElementA::DataType* A,
    const ElementB::DataType* B,
    int M,
    int N,
    int K,
    ElementC* C,
    ElementD* D,
    const ElementA::ScaleFactorType* SFA,
    const ElementB::ScaleFactorType* SFB,
    float scale,
    float beta) {
  using Gemm = typename KernelTraits::Gemm;
  using StrideA = typename KernelTraits::StrideA;
  using LayoutSFA = typename KernelTraits::LayoutSFA;
  using StrideB = typename KernelTraits::StrideB;
  using LayoutSFB = typename KernelTraits::LayoutSFB;
  using StrideC = typename KernelTraits::StrideC;
  using StrideD = typename KernelTraits::StrideD;
  using Sm1xxBlkScaledConfig = typename KernelTraits::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  auto problem_shape = cute::make_shape(M, N, K, 1);
  LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape);
  LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape);

  Gemm gemm_op;
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          A, stride_A,
          B, stride_B,
          SFA, layout_SFA,
          SFB, layout_SFB,
      },
      {
          {scale, beta},
          C, stride_C,
          D, stride_D,
      }};

  return gemm_op(arguments);
}

cutlass::Status dispatch_dense_variant(
    DenseKernelVariant variant,
    const ElementA::DataType* A,
    const ElementB::DataType* B,
    int M,
    int N,
    int K,
    ElementC* C,
    ElementD* D,
    const ElementA::ScaleFactorType* SFA,
    const ElementB::ScaleFactorType* SFB,
    float scale,
    float beta) {
  switch (variant) {
    case DenseKernelVariant::kBaselineAuto:
      return launch_dense_variant<DenseKernelBaseline>(A, B, M, N, K, C, D, SFA, SFB, scale, beta);
#if defined(SHARQ_TARGET_SM100A)
    case DenseKernelVariant::kSmall1Sm128x128x256:
      return launch_dense_variant<DenseKernelSmall1Sm>(A, B, M, N, K, C, D, SFA, SFB, scale, beta);
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

std::optional<DenseKernelVariant> forced_dense_variant_from_env() {
  char const* value = std::getenv("SHARQ_DENSE_NVFP4_KERNEL");
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }

  std::string_view name(value);
  if (name == "baseline_auto") {
    return DenseKernelVariant::kBaselineAuto;
  }
#if defined(SHARQ_TARGET_SM100A)
  if (name == "small_1sm_128x128x256") {
    return DenseKernelVariant::kSmall1Sm128x128x256;
  }
#endif
  return std::nullopt;
}

char const* dense_variant_name(DenseKernelVariant variant) {
  switch (variant) {
    case DenseKernelVariant::kBaselineAuto:
      return "baseline_auto";
#if defined(SHARQ_TARGET_SM100A)
    case DenseKernelVariant::kSmall1Sm128x128x256:
      return "small_1sm_128x128x256";
#endif
  }
  return "unknown";
}

int dense_autotune_iterations(int M, int N, int K) {
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

DenseKernelVariant dense_heuristic_variant(int M, int N, int K) {
#if defined(SHARQ_TARGET_SM100A)
  if (M >= 32768 && N >= 13824 && K >= 4096) {
    return DenseKernelVariant::kBaselineAuto;
  }
  if (M <= 256) {
    return DenseKernelVariant::kSmall1Sm128x128x256;
  }
#else
  (void)M;
  (void)N;
  (void)K;
#endif
  return DenseKernelVariant::kBaselineAuto;
}

bool use_dense_baseline_large_wide_regime(int M, int N, int K) {
#if defined(SHARQ_TARGET_SM100A)
  return M >= 32768 && N >= 13824 && K >= 4096;
#else
  (void)M;
  (void)N;
  (void)K;
  return false;
#endif
}

float benchmark_dense_variant(
    DenseKernelVariant variant,
    const ElementA::DataType* A,
    const ElementB::DataType* B,
    int M,
    int N,
    int K,
    ElementC* C,
    ElementD* D,
    const ElementA::ScaleFactorType* SFA,
    const ElementB::ScaleFactorType* SFB,
    float scale,
    float beta) {
  constexpr int kWarmupIters = 2;
  constexpr int kTimingRepeats = 2;
  int iters = dense_autotune_iterations(M, N, K);
  float best_ms = std::numeric_limits<float>::infinity();

  for (int repeat = 0; repeat < kTimingRepeats; ++repeat) {
    for (int i = 0; i < kWarmupIters; ++i) {
      if (dispatch_dense_variant(variant, A, B, M, N, K, C, D, SFA, SFB, scale, beta) != cutlass::Status::kSuccess) {
        return std::numeric_limits<float>::infinity();
      }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
      if (dispatch_dense_variant(variant, A, B, M, N, K, C, D, SFA, SFB, scale, beta) != cutlass::Status::kSuccess) {
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        return std::numeric_limits<float>::infinity();
      }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    best_ms = std::min(best_ms, elapsed_ms / static_cast<float>(iters));
  }

  return best_ms;
}

DenseKernelVariant autotune_dense_variant(
    int device,
    const ElementA::DataType* A,
    const ElementB::DataType* B,
    int M,
    int N,
    int K,
    ElementC* C,
    ElementD* D,
    const ElementA::ScaleFactorType* SFA,
    const ElementB::ScaleFactorType* SFB,
    float scale,
    float beta) {
  DenseKernelVariant best_variant = dense_heuristic_variant(M, N, K);
  float best_ms = std::numeric_limits<float>::infinity();

  constexpr DenseKernelVariant kCandidates[] = {
      DenseKernelVariant::kBaselineAuto,
#if defined(SHARQ_TARGET_SM100A)
      DenseKernelVariant::kSmall1Sm128x128x256,
#endif
  };

  for (DenseKernelVariant candidate : kCandidates) {
    float candidate_ms = benchmark_dense_variant(candidate, A, B, M, N, K, C, D, SFA, SFB, scale, beta);
    if (candidate_ms < best_ms) {
      best_ms = candidate_ms;
      best_variant = candidate;
    }
  }

  if (!std::isfinite(best_ms)) {
    best_variant = dense_heuristic_variant(M, N, K);
  }

  if (env_flag_enabled("SHARQ_NVFP4_LOG_AUTOTUNE")) {
    std::cerr << "Dense NVFP4 autotune device=" << device
              << " M=" << M << " N=" << N << " K=" << K
              << " -> " << dense_variant_name(best_variant);
    if (std::isfinite(best_ms)) {
      std::cerr << " (" << best_ms << " ms)";
    }
    std::cerr << std::endl;
  }

  return best_variant;
}

DenseKernelVariant select_dense_variant(
    const ElementA::DataType* A,
    const ElementB::DataType* B,
    int M,
    int N,
    int K,
    ElementC* C,
    ElementD* D,
    const ElementA::ScaleFactorType* SFA,
    const ElementB::ScaleFactorType* SFB,
    float scale,
    float beta) {
  if (auto forced = forced_dense_variant_from_env()) {
    return *forced;
  }

  if (use_dense_baseline_large_wide_regime(M, N, K)) {
    return DenseKernelVariant::kBaselineAuto;
  }

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  DenseShapeKey key{device, M, N, K};

  {
    std::lock_guard<std::mutex> lock(g_dense_variant_cache_mutex);
    auto it = g_dense_variant_cache.find(key);
    if (it != g_dense_variant_cache.end()) {
      return it->second;
    }
  }

  DenseKernelVariant selected = env_flag_enabled("SHARQ_NVFP4_DISABLE_AUTOTUNE")
      ? dense_heuristic_variant(M, N, K)
      : autotune_dense_variant(device, A, B, M, N, K, C, D, SFA, SFB, scale, beta);

  {
    std::lock_guard<std::mutex> lock(g_dense_variant_cache_mutex);
    g_dense_variant_cache.emplace(key, selected);
  }
  return selected;
}

}  // namespace

void matmul_host_nvfp4_bf16(
    const ElementA::DataType* A,
    const ElementB::DataType* B,
    int M,
    int N,
    int K,
    ElementC* C,
    ElementD* D,
    const ElementA::ScaleFactorType* SFA,
    const ElementB::ScaleFactorType* SFB,
    float scale,
    float beta) {
  DenseKernelVariant variant = select_dense_variant(A, B, M, N, K, C, D, SFA, SFB, scale, beta);
  cutlass::Status status = dispatch_dense_variant(variant, A, B, M, N, K, C, D, SFA, SFB, scale, beta);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM operation in matmul_host_nvfp4_bf16 failed with status: "
              << cutlass::cutlassGetStatusString(status)
              << " (Enum value: " << static_cast<int>(status) << ")"
              << ", variant=" << dense_variant_name(variant)
              << std::endl;
  }
  assert(status == cutlass::Status::kSuccess);
}

