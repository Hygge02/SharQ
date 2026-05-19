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


class ProfileRecorder:
    def __init__(self):
        self.enabled = False
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.scopes: list[str] = []

    def reset(self):
        self.events.clear()

    @contextlib.contextmanager
    def scope(self, name: str):
        self.scopes.append(name)
        try:
            yield
        finally:
            self.scopes.pop()

    def in_scope(self, name: str) -> bool:
        return name in self.scopes

    @contextlib.contextmanager
    def range(self, category: str | tuple[str, ...] | list[str] | None, nvtx_name: str):
        with nvtx_range(nvtx_name):
            if not self.enabled or category is None:
                yield
                return
            categories = (category,) if isinstance(category, str) else tuple(category)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                for current_category in categories:
                    self.events.setdefault(current_category, []).append((start_event, end_event))

    def elapsed_ms(self, category: str) -> float:
        return sum(start.elapsed_time(end) for start, end in self.events.get(category, ()))


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


def _wrap_method(cls, method_name: str, category_fn, range_name_fn, recorder: ProfileRecorder):
    original = getattr(cls, method_name)
    if getattr(original, "_sharq_profile_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        category = category_fn(self, *args, **kwargs)
        if category is None:
            return original(self, *args, **kwargs)
        primary_category = category if isinstance(category, str) else category[0]
        with recorder.range(category, range_name_fn(primary_category, self, *args, **kwargs)):
            return original(self, *args, **kwargs)

    wrapped._sharq_profile_wrapped = True
    setattr(cls, method_name, wrapped)


def install_nvtx_wrappers(mode: str, recorder: ProfileRecorder):
    _patch_attention_blocks(mode, recorder)
    _patch_qlinear(mode, recorder)
    _patch_regular_linear(mode, recorder)
    _patch_rmsnorm(mode, recorder)
    _patch_sdpa(mode, recorder)


def _add_attention_block_class(attention_classes, module_name: str, class_name: str):
    try:
        module = __import__(module_name, fromlist=[class_name])
        attention_classes.append(getattr(module, class_name))
    except (AttributeError, ImportError):
        pass


def _patch_attention_blocks(mode: str, recorder: ProfileRecorder):
    attention_classes = []
    _add_attention_block_class(attention_classes, "qLlamaLayer", "QLlamaAttention")
    _add_attention_block_class(attention_classes, "qQwenLayer", "QQwen2Attention")
    _add_attention_block_class(attention_classes, "qMixtralLayer", "QMixtralAttention")
    _add_attention_block_class(attention_classes, "transformers.models.llama.modeling_llama", "LlamaAttention")
    _add_attention_block_class(attention_classes, "transformers.models.qwen2.modeling_qwen2", "Qwen2Attention")
    _add_attention_block_class(attention_classes, "transformers.models.mixtral.modeling_mixtral", "MixtralAttention")

    for attention_cls in attention_classes:
        original_forward = getattr(attention_cls, "forward", None)
        if original_forward is None or getattr(original_forward, "_sharq_profile_wrapped", False):
            continue

        @functools.wraps(original_forward)
        def wrapped_forward(self, *args, __original_forward=original_forward, **kwargs):
            with recorder.scope("attention_block"), recorder.range("attention_block_total", f"{mode}.attention_block"):
                return __original_forward(self, *args, **kwargs)

        wrapped_forward._sharq_profile_wrapped = True
        setattr(attention_cls, "forward", wrapped_forward)


def _patch_qlinear(mode: str, recorder: ProfileRecorder):
    from qLinearLayer import QLinearLayer

    def prepare_category(self, *args, **kwargs):
        categories = []
        if self.quant_type == "SHARQ":
            categories.append("sharq_kernel")
        elif self.quant_type == "NVFP4":
            categories.append("quantize_kernel")
        if categories and recorder.in_scope("attention_block"):
            categories.append("attention_linear_prepare")
        return tuple(categories) if categories else None

    def prepare_rmsnorm_category(self, *args, **kwargs):
        if self.quant_type == "SHARQ" and self.extra_fusion:
            categories = ["sharq_kernel"]
            if recorder.in_scope("attention_block"):
                categories.append("attention_linear_prepare")
            return tuple(categories)
        return None

    def apply_category(self, prepared, *args, **kwargs):
        tag = prepared[0]
        if tag in {"NVFP4", "SHARQ", "SHARQ_SIM", "HIF4_SIM", "SHARQ_HIF4_SIM"}:
            categories = ["gemm"]
            if recorder.in_scope("attention_block"):
                categories.append("attention_linear_gemm")
            return tuple(categories)
        return None

    _wrap_method(
        QLinearLayer,
        "prepare_input",
        prepare_category,
        lambda category, self, *args, **kwargs: f"{mode}.{category}",
        recorder,
    )
    _wrap_method(
        QLinearLayer,
        "prepare_input_rmsnorm",
        prepare_rmsnorm_category,
        lambda category, self, *args, **kwargs: f"{mode}.{category}",
        recorder,
    )
    _wrap_method(
        QLinearLayer,
        "apply_prepared",
        apply_category,
        lambda category, self, *args, **kwargs: f"{mode}.gemm",
        recorder,
    )


def _patch_regular_linear(mode: str, recorder: ProfileRecorder):
    original_forward = torch.nn.Linear.forward
    if getattr(original_forward, "_sharq_profile_wrapped", False):
        return

    @functools.wraps(original_forward)
    def wrapped_forward(self, *args, **kwargs):
        categories = ["gemm"]
        if recorder.in_scope("attention_block"):
            categories.append("attention_linear_gemm")
        with recorder.range(tuple(categories), f"{mode}.gemm"):
            return original_forward(self, *args, **kwargs)

    wrapped_forward._sharq_profile_wrapped = True
    torch.nn.Linear.forward = wrapped_forward


def _patch_rmsnorm(mode: str, recorder: ProfileRecorder):
    rmsnorm_classes = []

    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        rmsnorm_classes.append(LlamaRMSNorm)
    except ImportError:
        pass

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

        rmsnorm_classes.append(Qwen2RMSNorm)
    except ImportError:
        pass

    try:
        from transformers.models.mixtral.modeling_mixtral import MixtralRMSNorm

        rmsnorm_classes.append(MixtralRMSNorm)
    except ImportError:
        pass

    for rmsnorm_cls in rmsnorm_classes:
        _wrap_method(
            rmsnorm_cls,
            "forward",
            lambda self, *args, **kwargs: "rmsnorm",
            lambda category, self, *args, **kwargs: f"{mode}.rmsnorm",
            recorder,
        )


def _patch_sdpa(mode: str, recorder: ProfileRecorder):
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    if getattr(original_sdpa, "_sharq_profile_wrapped", False):
        return

    @functools.wraps(original_sdpa)
    def wrapped_sdpa(*args, **kwargs):
        with recorder.range("attention", f"{mode}.attention"):
            return original_sdpa(*args, **kwargs)

    wrapped_sdpa._sharq_profile_wrapped = True
    torch.nn.functional.scaled_dot_product_attention = wrapped_sdpa


def run_prefill_once(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    recorder: ProfileRecorder | None = None,
):
    range_ctx = recorder.range("prefill_total", "prefill.total") if recorder is not None else nvtx_range("prefill.total")
    with torch.inference_mode(), range_ctx:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)


def print_prefill_breakdown(mode: str, recorder: ProfileRecorder, wall_ms: float):
    total_ms = recorder.elapsed_ms("prefill_total")
    if total_ms <= 0.0:
        total_ms = wall_ms

    attention_ms, attention_overlap_ms = get_attention_excluding_linear_ms(recorder)

    measured = []
    accounted_ms = 0.0
    rows = [
        ("gemm", "GEMM", recorder.elapsed_ms("gemm")),
        ("sharq_kernel", "SharQ kernel", recorder.elapsed_ms("sharq_kernel")),
        ("quantize_kernel", "Quantize kernel", recorder.elapsed_ms("quantize_kernel")),
        ("attention_excl_linear", "Attention", attention_ms),
        ("rmsnorm", "RMSNorm", recorder.elapsed_ms("rmsnorm")),
    ]

    for category, label, elapsed_ms in rows:
        if category == "quantize_kernel" and mode != "NVFP4":
            continue
        if category == "sharq_kernel" and mode != "SHARQ":
            continue
        if elapsed_ms <= 0.0:
            continue
        measured.append((label, elapsed_ms))
        accounted_ms += elapsed_ms

    others_ms = total_ms - accounted_ms
    overlap_ms = 0.0
    if others_ms < 0.0:
        overlap_ms = -others_ms
        others_ms = 0.0

    measured.append(("Others", others_ms))

    print("Prefill CUDA event breakdown:")
    print(f"  {'Total':<18} {total_ms:10.3f} ms  {100.0:6.2f}%")
    for label, elapsed_ms in measured:
        pct = elapsed_ms / total_ms * 100.0 if total_ms > 0.0 else 0.0
        print(f"  {label:<18} {elapsed_ms:10.3f} ms  {pct:6.2f}%")
    print(f"  {'Wall time':<18} {wall_ms:10.3f} ms")
    if overlap_ms > 0.01:
        print(f"  Note: categorized ranges overlap total by {overlap_ms:.3f} ms.")
    if attention_overlap_ms > 0.01:
        print(f"  Note: attention linear subranges overlap attention block by {attention_overlap_ms:.3f} ms.")
    print_attention_block_detail(recorder, total_ms)


def get_attention_excluding_linear_ms(recorder: ProfileRecorder) -> tuple[float, float]:
    block_ms = recorder.elapsed_ms("attention_block_total")
    if block_ms <= 0.0:
        return recorder.elapsed_ms("attention"), 0.0

    proj_total_ms = recorder.elapsed_ms("attention_linear_gemm") + recorder.elapsed_ms("attention_linear_prepare")
    attention_ms = block_ms - proj_total_ms
    if attention_ms < 0.0:
        return 0.0, -attention_ms
    return attention_ms, 0.0


def print_attention_block_detail(recorder: ProfileRecorder, total_ms: float):
    block_ms = recorder.elapsed_ms("attention_block_total")
    if block_ms <= 0.0:
        return

    proj_gemm_ms = recorder.elapsed_ms("attention_linear_gemm")
    proj_prepare_ms = recorder.elapsed_ms("attention_linear_prepare")
    proj_total_ms = proj_gemm_ms + proj_prepare_ms
    excl_proj_ms, overlap_ms = get_attention_excluding_linear_ms(recorder)
    sdpa_ms = recorder.elapsed_ms("attention")

    def pct(value: float, denominator: float) -> float:
        return value / denominator * 100.0 if denominator > 0.0 else 0.0

    print("Attention block detail:")
    print(f"  {'Block total':<24} {block_ms:10.3f} ms  {pct(block_ms, total_ms):6.2f}% of prefill")
    print(f"  {'Linear proj GEMM':<24} {proj_gemm_ms:10.3f} ms  {pct(proj_gemm_ms, block_ms):6.2f}% of block")
    if proj_prepare_ms > 0.0:
        print(f"  {'Linear proj prepare':<24} {proj_prepare_ms:10.3f} ms  {pct(proj_prepare_ms, block_ms):6.2f}% of block")
    print(f"  {'Excl. linear proj':<24} {excl_proj_ms:10.3f} ms  {pct(excl_proj_ms, block_ms):6.2f}% of block")
    print(f"  {'SDPA inside':<24} {sdpa_ms:10.3f} ms  {pct(sdpa_ms, block_ms):6.2f}% of block")
    if overlap_ms > 0.01:
        print(f"  Note: attention subranges overlap block total by {overlap_ms:.3f} ms.")


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

    recorder = ProfileRecorder()
    install_nvtx_wrappers(args.mode, recorder)

    model = load_benchmark_model(
        args.model,
        mode=args.mode,
        device=device,
        kv_cache=args.kv_cache,
        extra_fusion=not args.no_extra_fusion,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    bench_inputs = build_benchmark_inputs(
        vocab_size=int(getattr(model.config, "vocab_size", 151936)),
        batch_size=args.batch_size,
        prefill_seq_len=args.prefill_seq_len,
        decode_steps=None,
        device=device,
        seed=args.seed,
    )

    for _ in range(args.warmup_steps):
        run_prefill_once(model, bench_inputs.prefill_input_ids, bench_inputs.prefill_attention_mask, recorder)
        _sync()
        _cleanup()

    _sync()
    recorder.reset()
    recorder.enabled = True
    if args.capture_range:
        _cuda_profiler_start()
    start = time.perf_counter()
    run_prefill_once(model, bench_inputs.prefill_input_ids, bench_inputs.prefill_attention_mask, recorder)
    _sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    recorder.enabled = False
    if args.capture_range:
        _cuda_profiler_stop()

    print(f"Profiled {args.mode} prefill: {elapsed_ms:.3f} ms")
    print_prefill_breakdown(args.mode, recorder, elapsed_ms)


if __name__ == "__main__":
    main(build_parser().parse_args())
