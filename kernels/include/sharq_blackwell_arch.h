#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cute/tensor.hpp"

namespace sharq::blackwell_arch {

using namespace cute;

#if defined(SHARQ_TARGET_SM100A)

using DenseArchTag = cutlass::arch::Sm100;
using DenseTileShape = Shape<_256, _256, _256>;
using DenseClusterShape = Shape<_2, _4, _1>;

using SparseArchTag = cutlass::arch::Sm100;
using SparseTileShape = Shape<_256, _128, _256>;
using SparseClusterShape = Shape<_2, _1, _1>;
using SparseKernelSchedule = cutlass::gemm::KernelSparseTmaWarpSpecialized2SmNvf4Sm100;
using SparseEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2SmNvf4;

inline constexpr char kArchName[] = "sm100a";
inline constexpr char kBuildDirName[] = "build_cmake_sm100a";

#elif defined(SHARQ_TARGET_SM120A)

using DenseArchTag = cutlass::arch::Sm120;
using DenseTileShape = Shape<_128, _128, _128>;
using DenseClusterShape = Shape<_1, _1, _1>;

using SparseArchTag = cutlass::arch::Sm120;
using SparseTileShape = Shape<_128, _128, _256>;
using SparseClusterShape = Shape<_1, _1, _1>;
using SparseKernelSchedule = cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
using SparseEpilogueSchedule = cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120;

inline constexpr char kArchName[] = "sm120a";
inline constexpr char kBuildDirName[] = "build_cmake_sm120a";

#else

#error "SharQ Blackwell kernels require SHARQ_TARGET_SM100A or SHARQ_TARGET_SM120A."

#endif

}  // namespace sharq::blackwell_arch
