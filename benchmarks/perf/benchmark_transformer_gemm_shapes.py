from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
FP8_BLOCK_SIZE = 32
FP8_MAX_POS = float(torch.finfo(torch.float8_e4m3fn).max)


def load_sharq_ops():
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from sharq_loader import load_sharq_ops as _load_sharq_ops

    return _load_sharq_ops(repo_root=REPO_ROOT)


@dataclass(frozen=True)
class ShapeCase:
    name: str
    family: str
    m: int
    n: int
    k: int
    notes: str


SHAPE_CASES = (
    ShapeCase("llama3_8b_ctx8k_attn", "llm", 8192, 4096, 4096, "8K prefill, attention/output proj"),
    ShapeCase("llama3_8b_ctx8k_mlp_up", "llm", 8192, 14336, 4096, "8K prefill, FFN up/gate proj"),
    ShapeCase("llama3_8b_ctx16k_attn", "llm", 16384, 4096, 4096, "16K prefill, attention/output proj"),
    ShapeCase("llama3_8b_ctx16k_mlp_up", "llm", 16384, 14336, 4096, "16K prefill, FFN up/gate proj"),
    ShapeCase("llama3_8b_ctx32k_attn", "llm", 32768, 4096, 4096, "32K prefill, attention/output proj"),
    ShapeCase("llama3_8b_ctx32k_mlp_up", "llm", 32768, 14336, 4096, "32K prefill, FFN up/gate proj"),
    ShapeCase("qwen2_7b_ctx8k_attn", "llm", 8192, 3584, 3584, "8K prefill, attention/output proj"),
    ShapeCase("qwen2_7b_ctx8k_mlp_up", "llm", 8192, 18944, 3584, "8K prefill, FFN up/gate proj"),
    ShapeCase("qwen2_7b_ctx16k_attn", "llm", 16384, 3584, 3584, "16K prefill, attention/output proj"),
    ShapeCase("qwen2_7b_ctx16k_mlp_up", "llm", 16384, 18944, 3584, "16K prefill, FFN up/gate proj"),
    ShapeCase("qwen2_7b_ctx32k_attn", "llm", 32768, 3584, 3584, "32K prefill, attention/output proj"),
    ShapeCase("qwen2_7b_ctx32k_mlp_up", "llm", 32768, 18944, 3584, "32K prefill, FFN up/gate proj"),
    ShapeCase("llama3_70b_ctx8k_attn", "llm", 8192, 8192, 8192, "8K prefill, attention/output proj"),
    ShapeCase("llama3_70b_ctx8k_mlp_up", "llm", 8192, 28672, 8192, "8K prefill, FFN up/gate proj"),
    ShapeCase("llama3_70b_ctx16k_attn", "llm", 16384, 8192, 8192, "16K prefill, attention/output proj"),
    ShapeCase("llama3_70b_ctx16k_mlp_up", "llm", 16384, 28672, 8192, "16K prefill, FFN up/gate proj"),
    ShapeCase("llama3_70b_ctx32k_attn", "llm", 32768, 8192, 8192, "32K prefill, attention/output proj"),
    ShapeCase("llama3_70b_ctx32k_mlp_up", "llm", 32768, 28672, 8192, "32K prefill, FFN up/gate proj"),
    ShapeCase("wan2_2_t2v_a14b_ctx8k_attn", "diffusion", 8192, 5120, 5120, "8K tokens, Wan2.2 T2V A14B DiT attention proj"),
    ShapeCase("wan2_2_t2v_a14b_ctx8k_mlp_up", "diffusion", 8192, 13824, 5120, "8K tokens, Wan2.2 T2V A14B DiT MLP up proj"),
    ShapeCase("wan2_2_t2v_a14b_ctx16k_attn", "diffusion", 16384, 5120, 5120, "16K tokens, Wan2.2 T2V A14B DiT attention proj"),
    ShapeCase("wan2_2_t2v_a14b_ctx16k_mlp_up", "diffusion", 16384, 13824, 5120, "16K tokens, Wan2.2 T2V A14B DiT MLP up proj"),
    ShapeCase("wan2_2_t2v_a14b_ctx32k_attn", "diffusion", 32768, 5120, 5120, "32K tokens, Wan2.2 T2V A14B DiT attention proj"),
    ShapeCase("wan2_2_t2v_a14b_ctx32k_mlp_up", "diffusion", 32768, 13824, 5120, "32K tokens, Wan2.2 T2V A14B DiT MLP up proj"),
)


def choose_iters(m: int, n: int, k: int) -> int:
    work = m * n * k
    if work >= 200_000_000_000:
        return 30
    if work >= 50_000_000_000:
        return 50
    if work >= 5_000_000_000:
        return 100
    return 200


def bench_cuda(fn, iters: int, device: torch.device, warmup: int = 40):
    with torch.cuda.device(device):
        out = None
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()
        torch.cuda.synchronize(device)
        return out, start.elapsed_time(end) / iters


def global_nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten().contiguous()


def quantize_fp8_blockwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype for FP8 blockwise quantization: {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("FP8 blockwise quantization requires a contiguous tensor")
    if x.shape[-1] % FP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP8 blockwise quantization requires the last dimension to be divisible by {FP8_BLOCK_SIZE}, got {x.shape[-1]}"
        )

    orig_shape = x.shape
    x_blocks = x.reshape(*orig_shape[:-1], orig_shape[-1] // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE)
    max_abs = torch.amax(torch.abs(x_blocks), dim=-1, keepdim=True).to(torch.float32)
    x_blocks_f32 = x_blocks.to(torch.float32)

    exponent_bias = 127
    descale = max_abs / FP8_MAX_POS
    exponent = torch.where(
        torch.isnan(descale),
        torch.full_like(descale, 0xFF, dtype=torch.uint8),
        (
            torch.clamp(
                torch.ceil(torch.log2(descale)),
                min=-exponent_bias,
                max=exponent_bias,
            )
            + exponent_bias
        ).to(torch.uint8),
    )

    descale_fp = torch.where(
        exponent == 0,
        torch.ones_like(max_abs),
        torch.exp2(exponent_bias - exponent.to(torch.float32)),
    )

    data_lp = torch.clamp(x_blocks_f32 * descale_fp, min=-FP8_MAX_POS, max=FP8_MAX_POS)
    data_lp = data_lp.to(torch.float8_e4m3fn).reshape(orig_shape).contiguous()

    scale_e8m0 = exponent.view(torch.float8_e8m0fnu).squeeze(-1)
    scale_packed = to_blocked(scale_e8m0)
    return data_lp, scale_packed


def make_col_major(matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = matrix.shape
    out = torch.empty_strided((rows, cols), (1, rows), device=matrix.device, dtype=matrix.dtype)
    out.copy_(matrix)
    return out


def tflops_dense_equivalent(m: int, n: int, k: int, runtime_ms: float) -> float:
    return 2.0 * m * n * k / (runtime_ms * 1e-3) / 1e12


def tflops_sparse_actual(m: int, n: int, k: int, runtime_ms: float) -> float:
    return 1.0 * m * n * k / (runtime_ms * 1e-3) / 1e12


def benchmark_case(case: ShapeCase, sharq_ops, device: torch.device) -> dict[str, object]:
    x = torch.randn((case.m, case.k), device=device, dtype=torch.bfloat16)
    w = torch.randn((case.n, case.k), device=device, dtype=torch.bfloat16)

    nvfp4_input_scale = global_nvfp4_scale(x)
    nvfp4_weight_scale = global_nvfp4_scale(w)
    nvfp4_alpha = float((nvfp4_input_scale * nvfp4_weight_scale).item())

    qweight_shared, scale_w_sparse, scale_w_dense = sharq_ops.quantize_w32_shared(
        (w / nvfp4_weight_scale).contiguous()
    )
    a_comp, e, sfa_sparse, qx_res, scale_x_res = sharq_ops.fused_sparse_residual_quantize_x(
        (x / nvfp4_input_scale).contiguous(),
        case.n,
    )

    x_fp8, scale_x_fp8 = quantize_fp8_blockwise(x.contiguous())
    w_fp8_row, scale_w_fp8 = quantize_fp8_blockwise(w.contiguous())
    w_fp8_col = make_col_major(w_fp8_row.t())

    sparse_kernel = lambda: sharq_ops.sparse_matmul(
        a_comp,
        qweight_shared,
        e,
        sfa_sparse,
        scale_w_sparse,
        case.m,
        case.n,
        case.k,
        alpha=nvfp4_alpha,
    )
    dense_kernel = lambda: sharq_ops.matmul(qx_res, qweight_shared, scale_x_res, scale_w_dense, nvfp4_alpha)
    fp8_kernel = lambda: torch._scaled_mm(
        x_fp8,
        w_fp8_col,
        scale_x_fp8,
        scale_w_fp8,
        out_dtype=torch.bfloat16,
    )

    iters = choose_iters(case.m, case.n, case.k)
    sparse_out, sparse_ms = bench_cuda(sparse_kernel, iters, device)
    dense_out, dense_ms = bench_cuda(dense_kernel, iters, device)
    fp8_out, fp8_ms = bench_cuda(fp8_kernel, iters, device)

    return {
        "name": case.name,
        "family": case.family,
        "notes": case.notes,
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "iters": iters,
        "nvfp4_sparse_ms": sparse_ms,
        "nvfp4_dense_ms": dense_ms,
        "fp8_blockwise_ms": fp8_ms,
        "sparse_vs_nvfp4_dense_speedup": dense_ms / sparse_ms,
        "sparse_vs_fp8_blockwise_speedup": fp8_ms / sparse_ms,
        "fp8_blockwise_vs_nvfp4_dense_speedup": dense_ms / fp8_ms,
        "nvfp4_sparse_dense_equiv_tflops": tflops_dense_equivalent(case.m, case.n, case.k, sparse_ms),
        "nvfp4_sparse_actual_tflops": tflops_sparse_actual(case.m, case.n, case.k, sparse_ms),
        "nvfp4_dense_tflops": tflops_dense_equivalent(case.m, case.n, case.k, dense_ms),
        "fp8_blockwise_tflops": tflops_dense_equivalent(case.m, case.n, case.k, fp8_ms),
        "nvfp4_sparse_checksum_16": sparse_out.flatten()[:16].float().sum().item(),
        "nvfp4_dense_checksum_16": dense_out.flatten()[:16].float().sum().item(),
        "fp8_blockwise_checksum_16": fp8_out.flatten()[:16].float().sum().item(),
    }


def select_cases(families: set[str], names: set[str]) -> list[ShapeCase]:
    selected = []
    for case in SHAPE_CASES:
        if families and case.family not in families:
            continue
        if names and case.name not in names:
            continue
        selected.append(case)
    return selected


def print_result(result: dict[str, object]) -> None:
    print(f"{result['name']} [{result['family']}]")
    print(f"  shape: M={result['m']}, N={result['n']}, K={result['k']}")
    print(f"  note: {result['notes']}")
    print(f"  nvfp4_sparse_ms: {result['nvfp4_sparse_ms']:.6f}")
    print(f"  nvfp4_dense_ms: {result['nvfp4_dense_ms']:.6f}")
    print(f"  fp8_blockwise_ms: {result['fp8_blockwise_ms']:.6f}")
    print(f"  sparse_speedup_vs_nvfp4_dense: {result['sparse_vs_nvfp4_dense_speedup']:.4f}x")
    print(f"  sparse_speedup_vs_fp8_blockwise: {result['sparse_vs_fp8_blockwise_speedup']:.4f}x")
    print(f"  fp8_blockwise_speedup_vs_nvfp4_dense: {result['fp8_blockwise_vs_nvfp4_dense_speedup']:.4f}x")
    print(f"  nvfp4_sparse_dense_equiv_tflops: {result['nvfp4_sparse_dense_equiv_tflops']:.2f}")
    print(f"  nvfp4_sparse_actual_tflops: {result['nvfp4_sparse_actual_tflops']:.2f}")
    print(f"  nvfp4_dense_tflops: {result['nvfp4_dense_tflops']:.2f}")
    print(f"  fp8_blockwise_tflops: {result['fp8_blockwise_tflops']:.2f}")
    print(f"  checksum_sparse_16: {result['nvfp4_sparse_checksum_16']:.6f}")
    print(f"  checksum_nvfp4_dense_16: {result['nvfp4_dense_checksum_16']:.6f}")
    print(f"  checksum_fp8_blockwise_16: {result['fp8_blockwise_checksum_16']:.6f}")


def write_csv(results: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kernel-only benchmark for QLinear-aligned SHARQ NVFP4 sparse/dense kernels and FP8 block-wise GEMM."
    )
    parser.add_argument("--family", action="append", choices=["llm", "diffusion"], help="Filter to one or more shape families.")
    parser.add_argument("--case", action="append", help="Benchmark only the named case(s).")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sharq_ops = load_sharq_ops()
    families = set(args.family or [])
    names = set(args.case or [])
    cases = select_cases(families, names)
    if not cases:
        raise ValueError("No benchmark cases selected")

    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"capability: {torch.cuda.get_device_capability(device)}")
    print(f"selected_cases: {len(cases)}")
    print("comparison: qlinear-aligned sharq nvfp4 sparse kernel vs qlinear-aligned dense kernel vs fp8 block-wise kernel")

    results = []
    for case in cases:
        result = benchmark_case(case, sharq_ops, device)
        results.append(result)
        print_result(result)
        print()

    if args.csv:
        write_csv(results, args.csv)
        print(f"csv_written: {args.csv}")


if __name__ == "__main__":
    main()
