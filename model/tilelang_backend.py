from functools import lru_cache

import torch

try:
    import tilelang
    import tilelang.language as T
except Exception as exc:  # pragma: no cover - exercised only when optional dep is missing/broken
    tilelang = None
    T = None
    _TILELANG_IMPORT_ERROR = exc
else:
    _TILELANG_IMPORT_ERROR = None


NVFP4_GLOBAL_DENOM = 448.0 * 6.0
FP4_MAX = 6.0
UE4M3_MIN = 0.001953125
UE4M3_MAX_BYTE = 126


def _require_tilelang() -> None:
    if tilelang is None or T is None:
        raise ImportError(
            "SHARQ_TILELANG requires tilelang. Install tilelang==0.1.11 and "
            "apache-tvm-ffi==0.1.10 in this environment."
        ) from _TILELANG_IMPORT_ERROR


def global_nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.abs().max().float() / NVFP4_GLOBAL_DENOM, min=1e-9)


def _as_scale_1d(scale: torch.Tensor) -> torch.Tensor:
    return scale.detach().to(device=scale.device, dtype=torch.float32).reshape(1).contiguous()


def _check_2d_bf16_cuda(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be shaped [rows, cols], got {tuple(tensor.shape)}")


def _check_2d_uint8_cuda(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.uint8:
        raise ValueError(f"{name} must be torch.uint8, got {tensor.dtype}")
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be shaped [rows, cols], got {tuple(tensor.shape)}")


if T is not None:

    def _tl_clamp_fp4(x):
        return T.min(T.max(x, T.float32(-FP4_MAX)), T.float32(FP4_MAX))

    def _tl_e2m1_nibble(x):
        x = _tl_clamp_fp4(x)
        ax = T.abs(x)
        mag = T.if_then_else(
            ax <= T.float32(0.25),
            T.int32(0),
            T.if_then_else(
                ax < T.float32(0.75),
                T.int32(1),
                T.if_then_else(
                    ax <= T.float32(1.25),
                    T.int32(2),
                    T.if_then_else(
                        ax < T.float32(1.75),
                        T.int32(3),
                        T.if_then_else(
                            ax <= T.float32(2.5),
                            T.int32(4),
                            T.if_then_else(
                                ax < T.float32(3.5),
                                T.int32(5),
                                T.if_then_else(ax <= T.float32(5.0), T.int32(6), T.int32(7)),
                            ),
                        ),
                    ),
                ),
            ),
        )
        return T.if_then_else(x < T.float32(0.0), mag + T.int32(8), mag)

    def _tl_e2m1_value(nibble):
        nib = T.int32(nibble)
        sign = nib // T.int32(8)
        mag = nib - sign * T.int32(8)
        value = T.if_then_else(
            mag == 0,
            T.float32(0.0),
            T.if_then_else(
                mag == 1,
                T.float32(0.5),
                T.if_then_else(
                    mag == 2,
                    T.float32(1.0),
                    T.if_then_else(
                        mag == 3,
                        T.float32(1.5),
                        T.if_then_else(
                            mag == 4,
                            T.float32(2.0),
                            T.if_then_else(mag == 5, T.float32(3.0), T.if_then_else(mag == 6, T.float32(4.0), T.float32(6.0))),
                        ),
                    ),
                ),
            ),
        )
        return T.if_then_else(sign == 0, value, -value)

    def _tl_pack_nibbles(low, high):
        return T.uint8(low + high * T.int32(16))

    def _tl_low_nibble(byte):
        b = T.int32(byte)
        return b - (b // T.int32(16)) * T.int32(16)

    def _tl_high_nibble(byte):
        return T.int32(byte) // T.int32(16)

    def _tl_ue4m3_byte(x):
        y = T.min(T.max(x, T.float32(UE4M3_MIN)), T.float32(448.0))
        denorm_mant = T.int32(T.round(y * T.float32(512.0)))
        denorm_mant = T.min(T.max(denorm_mant, T.int32(1)), T.int32(8))

        exponent = T.floor(T.log2(y))
        exponent_scale = T.pow(T.float32(2.0), exponent)
        exp_bits = T.int32(exponent) + T.int32(7)
        mant = T.int32(T.round((y / exponent_scale - T.float32(1.0)) * T.float32(8.0)))
        carry = mant >= T.int32(8)
        exp_bits = exp_bits + T.if_then_else(carry, T.int32(1), T.int32(0))
        mant = T.if_then_else(carry, T.int32(0), mant)
        normal_byte = exp_bits * T.int32(8) + mant
        normal_byte = T.min(T.max(normal_byte, T.int32(1)), T.int32(UE4M3_MAX_BYTE))

        return T.uint8(T.if_then_else(y < T.float32(0.015625), denorm_mant, normal_byte))

    def _tl_ue4m3_value(byte):
        b = T.int32(byte)
        b = T.min(b, T.int32(UE4M3_MAX_BYTE))
        exp_bits = b // T.int32(8)
        mant = b - exp_bits * T.int32(8)
        denorm = T.float32(mant) * T.float32(UE4M3_MIN)
        normal = (T.float32(1.0) + T.float32(mant) / T.float32(8.0)) * T.pow(
            T.float32(2.0), T.float32(exp_bits - T.int32(7))
        )
        return T.if_then_else(exp_bits == 0, denorm, normal)

    def _tl_pair_score(x0, x1):
        return T.max(T.abs(x0), T.abs(x1))

    def _tl_i32(pred):
        return T.if_then_else(pred, T.int32(1), T.int32(0))

    def _tl_keep_pair0(s0, s1, s2, s3):
        rank = _tl_i32(s1 > s0) + _tl_i32(s2 > s0) + _tl_i32(s3 > s0)
        return rank < 2

    def _tl_keep_pair1(s0, s1, s2, s3):
        rank = _tl_i32(s0 >= s1) + _tl_i32(s2 > s1) + _tl_i32(s3 > s1)
        return rank < 2

    def _tl_keep_pair2(s0, s1, s2, s3):
        rank = _tl_i32(s0 >= s2) + _tl_i32(s1 >= s2) + _tl_i32(s3 > s2)
        return rank < 2

    def _tl_keep_pair3(s0, s1, s2, s3):
        rank = _tl_i32(s0 >= s3) + _tl_i32(s1 >= s3) + _tl_i32(s2 >= s3)
        return rank < 2

    def _tl_pair_keep(pair_idx, s0, s1, s2, s3):
        return T.if_then_else(
            pair_idx == 0,
            _tl_keep_pair0(s0, s1, s2, s3),
            T.if_then_else(
                pair_idx == 1,
                _tl_keep_pair1(s0, s1, s2, s3),
                T.if_then_else(pair_idx == 2, _tl_keep_pair2(s0, s1, s2, s3), _tl_keep_pair3(s0, s1, s2, s3)),
            ),
        )

    def _tl_load_packed_value(Q, row, elem):
        byte = Q[row, elem // 2]
        nibble = T.if_then_else((elem - (elem // 2) * 2) == 0, _tl_low_nibble(byte), _tl_high_nibble(byte))
        return _tl_e2m1_value(nibble)


@lru_cache(maxsize=128)
def _get_quantize_w32_kernel(n: int, k: int):
    _require_tilelang()
    if k % 32 != 0:
        raise ValueError(f"TileLang W32 quantization requires K % 32 == 0, got K={k}")

    @tilelang.jit(out_idx=[1, 2])
    def _compile(n_const: int, k_const: int):
        @T.prim_func
        def kernel(
            W: T.Tensor((n_const, k_const), "bfloat16"),
            Wq: T.Tensor((n_const, k_const // 2), "uint8"),
            Wsf: T.Tensor((n_const, k_const // 32), "uint8"),
        ):
            with T.Kernel(k_const // 32, n_const, threads=1) as (group_id, row):
                base = group_id * 32
                max_abs = T.alloc_var("float32", T.float32(0.0))
                for i in T.unroll(32):
                    max_abs = T.max(max_abs, T.abs(T.float32(W[row, base + i])))

                scale_byte = _tl_ue4m3_byte(max_abs / T.float32(FP4_MAX))
                scale = _tl_ue4m3_value(scale_byte)
                inv_scale = T.float32(1.0) / scale
                Wsf[row, group_id] = scale_byte

                for i in T.unroll(16):
                    q0 = _tl_e2m1_nibble(T.float32(W[row, base + 2 * i]) * inv_scale)
                    q1 = _tl_e2m1_nibble(T.float32(W[row, base + 2 * i + 1]) * inv_scale)
                    Wq[row, base // 2 + i] = _tl_pack_nibbles(q0, q1)

        return kernel

    return _compile(n, k)


@lru_cache(maxsize=128)
def _get_sparse_residual_quantize_kernel(m: int, k: int):
    _require_tilelang()
    if k % 128 != 0:
        raise ValueError(f"TileLang SharQ activation quantization requires K % 128 == 0, got K={k}")

    @tilelang.jit(out_idx=[1, 2, 3, 4])
    def _compile(m_const: int, k_const: int):
        @T.prim_func
        def kernel(
            X: T.Tensor((m_const, k_const), "bfloat16"),
            X_sparse_q: T.Tensor((m_const, k_const // 2), "uint8"),
            X_sparse_sf: T.Tensor((m_const, k_const // 32), "uint8"),
            X_res_q: T.Tensor((m_const, k_const // 2), "uint8"),
            X_res_sf: T.Tensor((m_const, k_const // 16), "uint8"),
        ):
            with T.Kernel(k_const // 32, m_const, threads=1) as (group_id, row):
                base = group_id * 32
                sparse_max = T.alloc_var("float32", T.float32(0.0))

                for chunk in T.unroll(4):
                    chunk_base = base + chunk * 8
                    v0 = T.float32(X[row, chunk_base + 0])
                    v1 = T.float32(X[row, chunk_base + 1])
                    v2 = T.float32(X[row, chunk_base + 2])
                    v3 = T.float32(X[row, chunk_base + 3])
                    v4 = T.float32(X[row, chunk_base + 4])
                    v5 = T.float32(X[row, chunk_base + 5])
                    v6 = T.float32(X[row, chunk_base + 6])
                    v7 = T.float32(X[row, chunk_base + 7])
                    s0 = _tl_pair_score(v0, v1)
                    s1 = _tl_pair_score(v2, v3)
                    s2 = _tl_pair_score(v4, v5)
                    s3 = _tl_pair_score(v6, v7)
                    for pair in T.unroll(4):
                        keep = _tl_pair_keep(pair, s0, s1, s2, s3)
                        a = T.float32(X[row, chunk_base + pair * 2])
                        b = T.float32(X[row, chunk_base + pair * 2 + 1])
                        sparse_max = T.if_then_else(keep, T.max(sparse_max, T.max(T.abs(a), T.abs(b))), sparse_max)

                sparse_scale_byte = _tl_ue4m3_byte(sparse_max / T.float32(FP4_MAX))
                sparse_scale = _tl_ue4m3_value(sparse_scale_byte)
                sparse_inv_scale = T.float32(1.0) / sparse_scale
                X_sparse_sf[row, group_id] = sparse_scale_byte

                for chunk in T.unroll(4):
                    chunk_base = base + chunk * 8
                    v0 = T.float32(X[row, chunk_base + 0])
                    v1 = T.float32(X[row, chunk_base + 1])
                    v2 = T.float32(X[row, chunk_base + 2])
                    v3 = T.float32(X[row, chunk_base + 3])
                    v4 = T.float32(X[row, chunk_base + 4])
                    v5 = T.float32(X[row, chunk_base + 5])
                    v6 = T.float32(X[row, chunk_base + 6])
                    v7 = T.float32(X[row, chunk_base + 7])
                    s0 = _tl_pair_score(v0, v1)
                    s1 = _tl_pair_score(v2, v3)
                    s2 = _tl_pair_score(v4, v5)
                    s3 = _tl_pair_score(v6, v7)
                    for pair in T.unroll(4):
                        elem0 = chunk_base + pair * 2
                        elem1 = elem0 + 1
                        keep = _tl_pair_keep(pair, s0, s1, s2, s3)
                        q0 = T.if_then_else(
                            keep,
                            _tl_e2m1_nibble(T.float32(X[row, elem0]) * sparse_inv_scale),
                            T.int32(0),
                        )
                        q1 = T.if_then_else(
                            keep,
                            _tl_e2m1_nibble(T.float32(X[row, elem1]) * sparse_inv_scale),
                            T.int32(0),
                        )
                        X_sparse_q[row, elem0 // 2] = _tl_pack_nibbles(q0, q1)

                for group16 in T.unroll(2):
                    group16_base = base + group16 * 16
                    residual_max = T.alloc_var("float32", T.float32(0.0))
                    for i in T.unroll(16):
                        elem = group16_base + i
                        sparse_q = _tl_load_packed_value(X_sparse_q, row, elem) * sparse_scale
                        residual = T.float32(X[row, elem]) - sparse_q
                        residual_max = T.max(residual_max, T.abs(residual))

                    residual_scale_byte = _tl_ue4m3_byte(residual_max / T.float32(FP4_MAX))
                    residual_scale = _tl_ue4m3_value(residual_scale_byte)
                    residual_inv_scale = T.float32(1.0) / residual_scale
                    X_res_sf[row, group_id * 2 + group16] = residual_scale_byte

                    for i in T.unroll(8):
                        elem0 = group16_base + 2 * i
                        elem1 = elem0 + 1
                        sparse0 = _tl_load_packed_value(X_sparse_q, row, elem0) * sparse_scale
                        sparse1 = _tl_load_packed_value(X_sparse_q, row, elem1) * sparse_scale
                        res0 = T.float32(X[row, elem0]) - sparse0
                        res1 = T.float32(X[row, elem1]) - sparse1
                        q0 = _tl_e2m1_nibble(res0 * residual_inv_scale)
                        q1 = _tl_e2m1_nibble(res1 * residual_inv_scale)
                        X_res_q[row, elem0 // 2] = _tl_pack_nibbles(q0, q1)

        return kernel

    return _compile(m, k)


@lru_cache(maxsize=128)
def _get_sparse_residual_matmul_kernel(m: int, n: int, k: int, block_m: int, block_n: int):
    _require_tilelang()

    @tilelang.jit(out_idx=[7])
    def _compile(m_const: int, n_const: int, k_const: int, block_m_const: int, block_n_const: int):
        @T.prim_func
        def kernel(
            X_sparse_q: T.Tensor((m_const, k_const // 2), "uint8"),
            X_sparse_sf: T.Tensor((m_const, k_const // 32), "uint8"),
            X_res_q: T.Tensor((m_const, k_const // 2), "uint8"),
            X_res_sf: T.Tensor((m_const, k_const // 16), "uint8"),
            Wq: T.Tensor((n_const, k_const // 2), "uint8"),
            Wsf: T.Tensor((n_const, k_const // 32), "uint8"),
            output_scale: T.Tensor((1,), "float32"),
            Y: T.Tensor((m_const, n_const), "bfloat16"),
        ):
            with T.Kernel(T.ceildiv(m_const, block_m_const), T.ceildiv(n_const, block_n_const), threads=256) as (
                block_row,
                block_col,
            ):
                tid = T.get_thread_binding(0)
                local_m = tid // block_n_const
                local_n = tid - local_m * block_n_const
                row = block_row * block_m_const + local_m
                col = block_col * block_n_const + local_n
                if (row < m_const) & (col < n_const) & (local_m < block_m_const):
                    acc = T.alloc_var("float32", T.float32(0.0))
                    for kk in T.serial(k_const):
                        x_sparse = _tl_load_packed_value(X_sparse_q, row, kk) * _tl_ue4m3_value(X_sparse_sf[row, kk // 32])
                        x_res = _tl_load_packed_value(X_res_q, row, kk) * _tl_ue4m3_value(X_res_sf[row, kk // 16])
                        w_val = _tl_load_packed_value(Wq, col, kk) * _tl_ue4m3_value(Wsf[col, kk // 32])
                        acc += (x_sparse + x_res) * w_val
                    Y[row, col] = T.bfloat16(acc * output_scale[0])

        return kernel

    return _compile(m, n, k, block_m, block_n)


@torch.no_grad()
def quantize_weight_shared_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_tilelang()
    _check_2d_bf16_cuda("weight", weight)
    n, k = weight.shape
    scale = global_nvfp4_scale(weight)
    weight_scaled = (weight / scale).to(torch.bfloat16).contiguous()
    kernel = _get_quantize_w32_kernel(int(n), int(k))
    weight_q, weight_sf = kernel(weight_scaled)
    return weight_q, weight_sf, scale


@torch.no_grad()
def quantize_activation_sparse_residual_nvfp4(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_tilelang()
    _check_2d_bf16_cuda("x", x)
    m, k = x.shape
    scale = global_nvfp4_scale(x)
    x_scaled = (x / scale).to(torch.bfloat16).contiguous()
    kernel = _get_sparse_residual_quantize_kernel(int(m), int(k))
    x_sparse_q, x_sparse_sf, x_res_q, x_res_sf = kernel(x_scaled)
    return x_sparse_q, x_sparse_sf, x_res_q, x_res_sf, scale


@torch.no_grad()
def matmul_sparse_residual(
    x_sparse_q: torch.Tensor,
    x_sparse_sf: torch.Tensor,
    x_res_q: torch.Tensor,
    x_res_sf: torch.Tensor,
    weight_q: torch.Tensor,
    weight_sf: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    *,
    block_m: int = 8,
    block_n: int = 32,
) -> torch.Tensor:
    _require_tilelang()
    _check_2d_uint8_cuda("x_sparse_q", x_sparse_q)
    _check_2d_uint8_cuda("x_sparse_sf", x_sparse_sf)
    _check_2d_uint8_cuda("x_res_q", x_res_q)
    _check_2d_uint8_cuda("x_res_sf", x_res_sf)
    _check_2d_uint8_cuda("weight_q", weight_q)
    _check_2d_uint8_cuda("weight_sf", weight_sf)
    if x_sparse_q.shape != x_res_q.shape:
        raise ValueError(f"x_sparse_q and x_res_q shape mismatch: {x_sparse_q.shape} vs {x_res_q.shape}")
    if x_sparse_q.shape[1] != weight_q.shape[1]:
        raise ValueError(f"K mismatch: x has {x_sparse_q.shape[1] * 2}, weight has {weight_q.shape[1] * 2}")
    if block_m * block_n != 256:
        raise ValueError("The current scalar TileLang matmul maps one output element per thread and expects block_m*block_n == 256")

    m = x_sparse_q.shape[0]
    k = x_sparse_q.shape[1] * 2
    n = weight_q.shape[0]
    if x_sparse_sf.shape != (m, k // 32):
        raise ValueError(f"x_sparse_sf must be shaped {(m, k // 32)}, got {tuple(x_sparse_sf.shape)}")
    if x_res_sf.shape != (m, k // 16):
        raise ValueError(f"x_res_sf must be shaped {(m, k // 16)}, got {tuple(x_res_sf.shape)}")
    if weight_sf.shape != (n, k // 32):
        raise ValueError(f"weight_sf must be shaped {(n, k // 32)}, got {tuple(weight_sf.shape)}")

    kernel = _get_sparse_residual_matmul_kernel(int(m), int(n), int(k), int(block_m), int(block_n))
    return kernel(
        x_sparse_q.contiguous(),
        x_sparse_sf.contiguous(),
        x_res_q.contiguous(),
        x_res_sf.contiguous(),
        weight_q.contiguous(),
        weight_sf.contiguous(),
        _as_scale_1d(scale_x * scale_w),
    )


__all__ = [
    "global_nvfp4_scale",
    "matmul_sparse_residual",
    "quantize_activation_sparse_residual_nvfp4",
    "quantize_weight_shared_nvfp4",
]
