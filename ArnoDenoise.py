import math

import os

from typing import Optional


import numpy as np

import tifffile as tiff


from skimage.util import img_as_float32


try:

    from skimage.restoration import estimate_sigma as _sk_est_sigma


    _HAS_SK_EST = True

except Exception:

    _HAS_SK_EST = False


from bm3d import bm3d, BM3DStages




def _raised_cosine(n: int) -> np.ndarray:

    """1D half-cosine that rises from 0→1 across n samples (n==0 → empty)."""

    if n <= 0:

        return np.zeros((0,), dtype=np.float32)

    # t in (0..1]; avoid exactly 0 weight when overlapping (keeps blends smooth).

    t = (np.arange(n, dtype=np.float32) + 0.5) / n

    return 0.5 * (1.0 - np.cos(np.pi * t))




def _tile_weight(h: int, w: int, top_ov: int, bot_ov: int, left_ov: int, right_ov: int) -> np.ndarray:

    wy = np.ones((h,), dtype=np.float32)

    wx = np.ones((w,), dtype=np.float32)

    if top_ov > 0:

        wy[:top_ov] = _raised_cosine(top_ov)

    if bot_ov > 0:

        wy[-bot_ov:] = _raised_cosine(bot_ov)[::-1]

    if left_ov > 0:

        wx[:left_ov] = _raised_cosine(left_ov)

    if right_ov > 0:

        wx[-right_ov:] = _raised_cosine(right_ov)[::-1]

    return wy[:, None] * wx[None, :]




def _to_float01(a: np.ndarray) -> np.ndarray:

    a = np.asarray(a)

    if np.issubdtype(a.dtype, np.integer):

        maxv = 65535.0 if a.dtype == np.uint16 else float(np.iinfo(a.dtype).max)

        return (a.astype(np.float32) / maxv).astype(np.float32)

    return a.astype(np.float32, copy=False)




def _grad_energy(x: np.ndarray) -> float:

    # simple gradient-energy proxy for edge content

    gx = np.diff(x, axis=1)

    gy = np.diff(x, axis=0)

    return 0.5 * (gx.var() + gy.var())




def change_metrics(orig: np.ndarray, den: np.ndarray, eps_u16: int = 2) -> dict:

    """

    Measure how much `den` differs from `orig` (both 2D, same shape).

    Returns a dict of interpretable metrics.

    Simple function from ChatGPT. It might give you some reference value for your report.

    """

    of = _to_float01(orig)

    df = _to_float01(den)

    res = of - df


    mae = float(np.mean(np.abs(res)))

    rmse = float(np.sqrt(np.mean(res**2)))

    std = float(np.std(res))

    mad = float(np.median(np.abs(res - np.median(res))) * 1.4826)  # robust sigma


    # What fraction of pixels changed more than eps (on 16-bit scale)?

    if orig.dtype == np.uint16 and den.dtype == np.uint16:

        changed_pct = float(np.mean(np.abs(orig.astype(np.int32) - den.astype(np.int32)) > eps_u16))

    else:

        eps = eps_u16 / 65535.0

        changed_pct = float(np.mean(np.abs(of - df) > eps))


    # Edge/sharpness retention: after / before (≈1.0 means edges preserved)

    e_before = _grad_energy(of) + 1e-12

    e_after = _grad_energy(df)

    sharpness_retention = float(e_after / e_before)


    # Noise estimate (no-ref) pre/post (if scikit-image available)

    if _HAS_SK_EST:

        try:

            sig_before = float(_sk_est_sigma(of, channel_axis=None))

            sig_after = float(_sk_est_sigma(df, channel_axis=None))

            noise_reduction_db = float(20.0 * np.log10((sig_before + 1e-12) / (sig_after + 1e-12)))

        except Exception:

            sig_before = sig_after = noise_reduction_db = float("nan")

    else:

        sig_before = sig_after = noise_reduction_db = float("nan")


    # Residual energy relative to original energy (smaller is “lighter touch”)

    energy_ratio = float(np.sum(res**2) / (np.sum(of**2) + 1e-12))


    # Focus on flat regions only (to see “pure noise” removal)

    gx = np.pad(np.abs(np.diff(of, axis=1)), ((0, 0), (0, 1)))

    gy = np.pad(np.abs(np.diff(of, axis=0)), ((0, 1), (0, 0)))

    grad = np.hypot(gx, gy)

    th = np.quantile(grad, 0.25)  # bottom 25% gradients

    flat = grad <= th

    flat_mae = float(np.mean(np.abs(res[flat]))) if flat.any() else float("nan")


    return dict(

        mae=mae,

        rmse=rmse,

        std=std,

        mad=mad,

        changed_pct=changed_pct,

        sharpness_retention=sharpness_retention,

        noise_sigma_before=sig_before,

        noise_sigma_after=sig_after,

        noise_reduction_db=noise_reduction_db,

        residual_energy_ratio=energy_ratio,

        flat_mae=flat_mae,

    )




def _tile_weight(h: int, w: int, top_ov: int, bot_ov: int, left_ov: int, right_ov: int) -> np.ndarray:

    """Separable raised-cosine weight that is 1.0 in the core and ramps across overlaps."""

    wy = np.ones((h,), dtype=np.float32)

    wx = np.ones((w,), dtype=np.float32)


    if top_ov > 0:

        wy[:top_ov] = _raised_cosine(top_ov)

    if bot_ov > 0:

        wy[-bot_ov:] = _raised_cosine(bot_ov)[::-1]

    if left_ov > 0:

        wx[:left_ov] = _raised_cosine(left_ov)

    if right_ov > 0:

        wx[-right_ov:] = _raised_cosine(right_ov)[::-1]


    return wy[:, None] * wx[None, :]




def _bm3d_full(tile_f32: np.ndarray, sigma: float) -> np.ndarray:

    """

    High-quality BM3D: basic (HT) + Wiener stage using pilot as ref.

    Falls back to ALL_STAGES if the bm3d version doesn't support 'ref'.

    """

    if sigma <= 0:

        return tile_f32  # nothing to do

    # I did some experiments here with doing it in two stages. That....did not work properly.

    # Stage 1: basic estimate (hard-thresholding)

    # basic = bm3d(tile_f32, sigma_psd=sigma, stage_arg=BM3DStages.HARD_THRESHOLDING)

    # # Stage 2: Wiener filtering with the basic estimate as the pilot

    # try:

    #     final = bm3d(tile_f32, sigma_psd=sigma, stage_arg=basic)

    # except TypeError:

    final = bm3d(tile_f32, sigma_psd=sigma, stage_arg=BM3DStages.ALL_STAGES)

    return final




def _estimate_sigma_global(img_f32: np.ndarray) -> float:

    """Sigma in [0,1] intensity units. Uses skimage if available, else a sane default."""

    if _HAS_SK_EST:

        try:

            return float(_sk_est_sigma(img_f32, channel_axis=None))

        except Exception:

            pass

    # Fallback. My estimations show that this should be appropriate for the 4µs dwell time images. Feel free to change it.

    return 1.5e-3  # ≈ 0.0015 on [0,1]




def _estimate_sigma_flat(patch_f32: np.ndarray) -> float:

    # Robust gradient mask

    gx = np.pad(np.abs(np.diff(patch_f32, axis=1)), ((0, 0), (0, 1)))

    gy = np.pad(np.abs(np.diff(patch_f32, axis=0)), ((0, 1), (0, 0)))

    grad = np.hypot(gx, gy)

    th = np.quantile(grad, 0.25)  # bottom 25% gradients

    flat = grad <= th

    if flat.sum() < 256:  # too few flat pixels → fallback

        return float(np.median(np.abs(patch_f32 - np.median(patch_f32))) * 1.4826)

    # robust sigma on flat area

    med = np.median(patch_f32[flat])

    mad = np.median(np.abs(patch_f32[flat] - med))

    return float(1.4826 * mad)




def _anscombe_forward_counts(counts: np.ndarray) -> np.ndarray:

    """Forward Anscombe for Poisson counts: z = 2*sqrt(y + 3/8)."""

    return 2.0 * np.sqrt(counts.astype(np.float32) + 0.375, dtype=np.float32)




def _anscombe_inverse_simple(z: np.ndarray) -> np.ndarray:

    """Approximate inverse: y ≈ (z/2)^2 - 3/8 (fast, slightly biased at very low counts)."""

    return np.maximum((0.25 * (z.astype(np.float32) ** 2)) - 0.375, 0.0, dtype=np.float32)




def bm3d_denoise_2x2_cpu(

    im: np.ndarray,

    overlap: int = 256,

    use_vst: bool = True,

    sigma: Optional[float] = None,

    sigma_mult: float = 1.0,

    per_tile_sigma: bool = True,

    threads: Optional[int] = None,

    return_float: bool = False,

) -> np.ndarray:

    """

    CPU-only BM3D denoiser with 2×2 tiling and raised-cosine blending.


    Parameters

    ----------

    im : np.ndarray

        2D grayscale image. uint16 (preferred) or float in [0,1].

    overlap : int

        Overlap (pixels) on each internal tile edge. 128 is a good quality default.

    sigma : float or None

        Noise std-dev in [0,1] intensity units for BM3D. If None, it is estimated.

    sigma_mult : float

        Multiplier applied to (global or per-tile) sigma estimate.

    per_tile_sigma : bool

        If True and sigma is None, estimate sigma on each tile patch (best if noise varies).

        If False or sigma given, uses a single global sigma for all tiles.

    threads : int or None

        If set, constrains OMP thread count for NumPy/BLAS backends (can reduce thrash).

    return_float : bool

        If True → return float32 in [0,1]. Otherwise returns uint16.


    Returns

    -------

    np.ndarray

        Denoised image with original shape; dtype per `return_float`.

    """

    if threads is not None:

        os.environ["OMP_NUM_THREADS"] = str(int(threads))


    if im.ndim != 2:

        raise ValueError("Expected a 2D grayscale image.")


    H, W = im.shape


    # Normalize to float32 [0,1]

    if np.issubdtype(im.dtype, np.integer):

        scale = np.float32(65535.0 if im.dtype == np.uint16 else np.iinfo(im.dtype).max)

        f32 = img_as_float32(im.astype(np.float32) / scale)

    else:

        f32 = img_as_float32(im)


    # ---- Prepare domain ----

    if use_vst:

        # Treat input as *counts*

        if np.issubdtype(im.dtype, np.integer):

            counts = im.astype(np.float32)

            max_counts = float(np.iinfo(im.dtype).max)

        else:

            # float assumed in [0,1] → map to uint16-like counts

            counts = np.clip(im.astype(np.float32), 0.0, 1.0) * 65535.0

            max_counts = 65535.0


        # Forward VST

        z = _anscombe_forward_counts(counts)


        # Normalize to ~[0,1] using a *constant* scale (avoids image-dependent sigma)

        z_scale = 2.0 * math.sqrt(max_counts + 0.375)  # ≈ max possible z

        z_norm = (z / z_scale).astype(np.float32)


        # In VST domain, noise stdev ≈ 1 → after scaling: sigma ≈ 1/z_scale

        sigma_est = 1.0 / z_scale

    else:

        # No VST: just normalize to [0,1] if integer input

        if np.issubdtype(im.dtype, np.integer):

            z_norm = (im.astype(np.float32) / float(np.iinfo(im.dtype).max)).astype(np.float32)

        else:

            z_norm = np.clip(im.astype(np.float32), 0.0, 1.0)


        # Precompute global sigma if needed

        if sigma is None:

            sigma_est = _estimate_sigma_global(f32)

        else:

            sigma_est = float(0.0015)

        # sigma_est = 0.0015


    sigma1 = float(sigma_mult) * sigma_est


    # Fixed 2×2 grid core sizes

    core_h = math.ceil(H / 2)

    core_w = math.ceil(W / 2)


    out_accum = np.zeros((H, W), dtype=np.float32)

    wgt_accum = np.zeros((H, W), dtype=np.float32)


    # Iterate tiles (iy, ix) ∈ {0,1}×{0,1}

    for iy in range(2):

        for ix in range(2):

            y0 = iy * core_h

            x0 = ix * core_w

            ch = min(core_h, H - y0)

            cw = min(core_w, W - x0)


            # Expand with overlap but clamp to image bounds

            ys = max(0, y0 - overlap)

            xs = max(0, x0 - overlap)

            ye = min(H, y0 + ch + overlap)

            xe = min(W, x0 + cw + overlap)


            patch = f32[ys:ye, xs:xe]


            # Determine how much of this patch is actually overlapping on each side

            top_ov = y0 - ys  # >0 if we extended upward

            left_ov = x0 - xs  # >0 if we extended left

            bot_ov = ye - (y0 + ch)  # >0 if we extended downward

            right_ov = xe - (x0 + cw)  # >0 if we extended right


            # Choose sigma for this patch

            if (sigma is None and per_tile_sigma) and not use_vst:

                # sig = _estimate_sigma_global(patch) * float(sigma_mult)

                sig = _estimate_sigma_flat(patch)

                sig_1 = sig * sigma_mult


            else:

                sig_1 = sigma1


            # BM3D on the (overlapped) patch

            # den = _bm3d_full(patch, sigma=sig).astype(np.float32, copy=False)

            den = _bm3d_full(patch, sigma=sig_1).astype(np.float32, copy=False)


            # Smooth blending weights (1.0 in the core, raised-cosine across overlaps)

            w = _tile_weight(den.shape[0], den.shape[1], top_ov, bot_ov, left_ov, right_ov)


            # Accumulate

            out_accum[ys:ye, xs:xe] += den * w

            wgt_accum[ys:ye, xs:xe] += w


    # Normalize by weights (safe divide)

    mask = wgt_accum > 0

    z_den_norm = np.zeros_like(out_accum, dtype=np.float32)

    z_den_norm[mask] = out_accum[mask] / wgt_accum[mask]

    z_den_norm[~mask] = z_norm[~mask]  # safety


    if use_vst:

        z_den = (z_den_norm * z_scale).astype(np.float32)

        counts_hat = _anscombe_inverse_simple(z_den)  # fast inverse

        if return_float:

            # return float in [0,1] (relative to uint16 dynamic range)

            return np.clip(counts_hat / 65535.0, 0.0, 1.0).astype(np.float32)

        return np.clip(np.rint(counts_hat), 0.0, 65535.0).astype(np.uint16)

    else:

        if return_float:

            return np.clip(z_den_norm, 0.0, 1.0).astype(np.float32)

        return np.clip(np.rint(z_den_norm * 65535.0), 0.0, 65535.0).astype(np.uint16)




im = tiff.imread(r"D:\Project_data\MBSEM_noise\original_1pct.tiff")


# We don't have enough ram to run the full image so we split it in 4 sections with an overlap.

denoised = bm3d_denoise_2x2_cpu(

    im,

    overlap=256,

    use_vst=False,  # This doesn't work properly yet. I'll need to look at this and maybe use makitalo-foi here.

    sigma=None,  # let it estimate; or pass a sigma value for known noise

    sigma_mult=0.85,  # This is a multiplier for the sigma estimates. Keep this between 0.8 and 1.2. 0.8 is conservative. 1.2 is very agressive

    per_tile_sigma=True,  # best quality when noise varies across the field

    threads=12,  # Edit this based on your CPU. Your CPU has more threads than mine.

    return_float=False,  # True to keep float32 [0,1]

)


change = change_metrics(im, denoised)  # This generates some metrics since we don't have a ground truth.


print(change)

tiff.imwrite(r"D:\Project_data\MBSEM_noise\bm3d_test\bm3d_test_no_vst.tiff", denoised)
