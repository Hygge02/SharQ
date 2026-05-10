from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_e2e_sharq import build_benchmark_inputs  # noqa: E402
from modeling_sharq import load_benchmark_model  # noqa: E402


@contextlib.contextmanager
def nvtx_range(name: str):
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_profiler_start():
    if torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


def _wrap_method(cls, method_name: str, range_name_fn):
    original = getattr(cls, method_name)
    if getattr(original, "_sharq_nvtx_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        with nvtx_range(range_name_fn(self, *args, **kwargs)):
            return original(self, *args, **kwargs)

    wrapped._sharq_nvtx_wrapped = True
    setattr(cls, method_name, wrapped)


def install_nvtx_wrappers(mode: str):
    rmsnorm_classes = []

    try:
        from qLlamaLayer import QLlamaRMSNorm

        rmsnorm_classes.append(QLlamaRMSNorm)
    except ImportError:
        pass

    try:
        from qQwenLayer import QQwen2RMSNorm

        rmsnorm_classes.append(QQwen2RMSNorm)
    except ImportError:
        pass

    try:
        from qMixtralLayer import QMixtralRMSNorm

        rmsnorm_classes.append(QMixtralRMSNorm)
    except ImportError:
        pass

    for rmsnorm_cls in rmsnorm_classes:
        _wrap_method(
            rmsnorm_cls,
            "forward",
            lambda self, *args, **kwargs: f"{mode}.rmsnorm",
        )


def _patch_sdpa(mode: str):
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    if getattr(original_sdpa, "_sharq_nvtx_wrapped", False):
        return

    @functools.wraps(original_sdpa)
    def wrapped_sdpa(*args, **kwargs):
        with nvtx_range(f"{mode}.attention"):
            return original_sdpa(*args, **kwargs)

    wrapped_sdpa._sharq_nvtx_wrapped = True
    torch.nn.functional.scaled_dot_product_attention = wrapped_sdpa


def _patch_sharq_ops(mode: str):
    import quantize

    ops = quantize.load_sharq_ops()

    def wrap_op(name: str, range_name: str):
        original = getattr(ops, name, None)
        if original is None or getattr(original, "_sharq_nvtx_wrapped", False):
            return

        def wrapped(*args, **kwargs):
            with nvtx_range(range_name):
                return original(*args, **kwargs)

        wrapped._sharq_nvtx_wrapped = True
        setattr(ops, name, wrapped)

    if mode == "SHARQ":
        wrap_op("fused_sparse_residual_quantize_x", "SHARQ.sharq_kernel")
        wrap_op("fused_rmsnorm_sparse_residual_quantize_x", "SHARQ.sharq_kernel")
        wrap_op("sparse_matmul", "SHARQ.linear_gemm.sparse_nvfp4")
        wrap_op("sparse_matmul_tensor_scale", "SHARQ.linear_gemm.sparse_nvfp4")
        wrap_op("matmul_accum", "SHARQ.linear_gemm.dense_nvfp4")
        wrap_op("matmul_accum_tensor_scale", "SHARQ.linear_gemm.dense_nvfp4")
        wrap_op("matmul", "SHARQ.linear_gemm.dense_nvfp4")
        wrap_op("matmul_tensor_scale", "SHARQ.linear_gemm.dense_nvfp4")
    elif mode == "NVFP4":
        wrap_op("quantize_x_nvfp4", "NVFP4.quantize_kernel")
        wrap_op("matmul", "NVFP4.linear_gemm.dense_nvfp4")
        wrap_op("matmul_tensor_scale", "NVFP4.linear_gemm.dense_nvfp4")


def run_prefill_once(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    with torch.inference_mode(), nvtx_range("prefill.total"):
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Llama/Qwen/Mixtral prefill with NVTX ranges for SharQ/NVFP4.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, default="SHARQ", choices=["SHARQ", "NVFP4", "FP16"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prefill_seq_len", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kv_cache", action="store_true")
    parser.add_argument("--no_extra_fusion", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--capture_range",
        action="store_true",
        help="Use cudaProfilerStart/Stop around the measured prefill for nsys --capture-range=cudaProfilerApi.",
    )
    return parser


def main(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    install_nvtx_wrappers(args.mode)
    _patch_sdpa(args.mode)

    model = load_benchmark_model(
        args.model,
        mode=args.mode,
        device=device,
        kv_cache=args.kv_cache,
        extra_fusion=not args.no_extra_fusion,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    if args.mode in {"SHARQ", "NVFP4"}:
        _patch_sharq_ops(args.mode)

    bench_inputs = build_benchmark_inputs(
        vocab_size=int(getattr(model.config, "vocab_size", 151936)),
        batch_size=args.batch_size,
        prefill_seq_len=args.prefill_seq_len,
        decode_steps=None,
        device=device,
        seed=args.seed,
    )

    for _ in range(args.warmup_steps):
        run_prefill_once(model, bench_inputs.prefill_input_ids, bench_inputs.prefill_attention_mask)
        _sync()
        _cleanup()

    _sync()
    if args.capture_range:
        _cuda_profiler_start()
    start = time.perf_counter()
    run_prefill_once(model, bench_inputs.prefill_input_ids, bench_inputs.prefill_attention_mask)
    _sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if args.capture_range:
        _cuda_profiler_stop()

    print(f"Profiled {args.mode} prefill: {elapsed_ms:.3f} ms")
    print("Use Nsight Systems NVTX summaries for named ranges; compute others as prefill.total minus named categories.")


if __name__ == "__main__":
    main(build_parser().parse_args())
