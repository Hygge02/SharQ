from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def add_repo_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root, repo_root / "model"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


REPO_ROOT = add_repo_paths()

import tilelang_backend  # noqa: E402
from qLinearLayer import QLinearLayer  # noqa: E402
from quantize import global_nvfp4_scale, load_sharq_ops, quantize_nvfp4_tensor, top2_pairs_8_maxabs  # noqa: E402


@torch.no_grad()
def pseudo_sharq_tilelang(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    scale_x = global_nvfp4_scale(x)
    scale_w = global_nvfp4_scale(w)
    x_scaled = (x / scale_x).to(torch.bfloat16).float()
    w_scaled = (w / scale_w).to(torch.bfloat16).float()

    w_q32 = quantize_nvfp4_tensor(w_scaled, group_size=32)
    x_sparse = top2_pairs_8_maxabs(x_scaled)
    x_sparse_q32 = quantize_nvfp4_tensor(x_sparse, group_size=32)
    x_res_q16 = quantize_nvfp4_tensor(x_scaled - x_sparse_q32, group_size=16)
    return (F.linear(x_sparse_q32, w_q32) + F.linear(x_res_q16, w_q32)) * (scale_x * scale_w)


def summarize(name: str, pred: torch.Tensor, ref: torch.Tensor) -> None:
    diff = pred.float() - ref.float()
    print(name)
    print(f"  checksum [0:16]: {pred.flatten()[:16].float().sum().item():.8f}")
    print(f"  mean_abs       : {diff.abs().mean().item():.8f}")
    print(f"  max_abs        : {diff.abs().max().item():.8f}")


def compare_bytes(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    got = got.contiguous()
    ref = ref.contiguous()
    diff = got != ref
    print(name)
    print(f"  shape/dtype    : {tuple(got.shape)} {got.dtype}")
    print(f"  byte_mismatch  : {int(diff.sum().item())}")
    if diff.any():
        idx = diff.flatten().nonzero().flatten()[:8]
        got_flat = got.flatten()
        ref_flat = ref.flatten()
        pairs = [(int(i.item()), int(got_flat[i].item()), int(ref_flat[i].item())) for i in idx]
        print(f"  first_mismatch : {pairs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check the TileLang real-quant SharQ backend.")
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-native", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.k % 128 != 0:
        raise ValueError(f"k must be a multiple of 128, got {args.k}")
    if args.n % 32 != 0:
        raise ValueError(f"n should be a multiple of 32 for the default TileLang matmul tile, got {args.n}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    x = torch.randn((args.m, args.k), device="cuda", dtype=torch.bfloat16)
    layer = torch.nn.Linear(args.k, args.n, bias=False, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.2)

    qlinear = QLinearLayer(layer, quant_type="SHARQ_TILELANG")
    y_tilelang = qlinear(x)
    y_pseudo = pseudo_sharq_tilelang(x, layer.weight)
    y_ref = F.linear(x.float(), layer.weight.float())

    x_sparse_q, x_sparse_sf, x_res_q, x_res_sf, scale_x = tilelang_backend.quantize_activation_sparse_residual_nvfp4(x)
    w_q, w_sf, scale_w = tilelang_backend.quantize_weight_shared_nvfp4(layer.weight)

    torch.cuda.synchronize()

    print(f"problem: M={args.m}, N={args.n}, K={args.k}, seed={args.seed}")
    print(f"activation realquant: sparse_q={tuple(x_sparse_q.shape)} {x_sparse_q.dtype}, "
          f"sparse_sf={tuple(x_sparse_sf.shape)} {x_sparse_sf.dtype}, "
          f"res_q={tuple(x_res_q.shape)} {x_res_q.dtype}, res_sf={tuple(x_res_sf.shape)} {x_res_sf.dtype}")
    print(f"weight realquant    : q={tuple(w_q.shape)} {w_q.dtype}, sf={tuple(w_sf.shape)} {w_sf.dtype}")
    print(f"global scales       : x={float(scale_x.cpu()):.10f}, w={float(scale_w.cpu()):.10f}")
    print()
    summarize("TileLang realquant SharQ vs pseudo", y_tilelang, y_pseudo)
    print()
    summarize("TileLang realquant SharQ vs BF16", y_tilelang, y_ref)

    if args.compare_native:
        sharq_ops = load_sharq_ops()
        native_w_q, native_w_sf_sparse, _native_w_sf_dense = sharq_ops.quantize_w32_shared(
            (layer.weight / scale_w).to(torch.bfloat16).contiguous()
        )
        native_x_sparse_q, native_x_sparse_sf, native_x_res_q, native_x_res_sf = sharq_ops.fused_sparse_residual_quantize_x_debug(
            (x / scale_x).to(torch.bfloat16).contiguous(),
            args.n,
        )
        torch.cuda.synchronize()
        print()
        compare_bytes("weight payload vs native", w_q, native_w_q)
        compare_bytes("activation sparse payload vs native debug", x_sparse_q, native_x_sparse_q)
        compare_bytes("activation residual payload vs native debug", x_res_q, native_x_res_q)
        print("scale payloads are real UE4M3 bytes; TileLang stores them in compact row-major arrays, "
              "while native SHARQ stores them in CUTLASS scale layouts.")


if __name__ == "__main__":
    main()
