import os
import glob
import math
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from scipy.stats import gamma as gamma_dist

IMG_DIR = r"D:\Datasets\tid2008\reference_images"
NUM_IMAGES = 25

ADD_NOISE_VARIANCE = 900.0
PATCH_M1 = 7
N_EIGEN = 8

DELTA = 0.99
MAX_ITERS = 8
TOL_SIGMA = 0.25 
MIN_KEEP = 500

DEBUG_ITER_LOG = False

# theta coefficients
THETA_TABLE = {
    5:  np.array([28.1839,  0.0436, -0.0674, -0.1561,  0.2159, -0.0686,  0.0079, -0.1645,  0.0646]),
    10: np.array([108.7873, -0.0343,  0.0154, -0.0408, -0.0355,  0.0031, -0.0453,  0.0436,  0.0039]),
    15: np.array([225.6460, -0.0189,  0.0142,  0.0130, -0.0088,  0.0009, -0.0091,  0.0050,  0.0009]),
    20: np.array([399.8086,  0.0065, -0.0038, -0.0069, -0.0039,  0.0094, -0.0026,  0.0061, -0.0043]),
    25: np.array([625.1615, -0.0004,  0.0043,  0.0060, -0.0087, -0.0028,  0.0111, -0.0159,  0.0061]),
    30: np.array([900.0040, -0.0013,  0.0026, -0.0094,  0.0099,  0.0021, -0.0042,  0.0022, -0.0019]),
}
TRAIN_SIGMAS = np.array(sorted(THETA_TABLE.keys()))


def load_rgb_uint8(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)

def add_gaussian_noise_uint8(img_u8, var):
    sigma = math.sqrt(var)
    noise = np.random.normal(0.0, sigma, img_u8.shape).astype(np.float32)
    noisy = img_u8.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def sliding_3d_overlapping_patches(img_u8, m1):
    H, W, C = img_u8.shape
    sH = H - m1 + 1
    sW = W - m1 + 1
    if sH <= 0 or sW <= 0:
        raise ValueError(f"Image too small for m1={m1}: got H={H}, W={W}")

    strides = img_u8.strides
    patches = np.lib.stride_tricks.as_strided(
        img_u8,
        shape=(sH, sW, m1, m1, C),
        strides=(strides[0], strides[1], strides[0], strides[1], strides[2]),
        writeable=False
    ).reshape(sH * sW, m1, m1, C)
    return patches

def rgb_to_gray_u8(img_u8):
    r = img_u8[..., 0].astype(np.float32)
    g = img_u8[..., 1].astype(np.float32)
    b = img_u8[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)

def compute_s1_map(img_gray_u8, m1):
    img = img_gray_u8.astype(np.float32)
    H, W = img.shape
    sH, sW = H - m1 + 1, W - m1 + 1
    if sH <= 0 or sW <= 0:
        raise ValueError(f"Image too small for m1={m1}")

    kx = np.array([[-1, 1]], dtype=np.float32)
    ky = kx.T

    gx = convolve(img, kx, mode="nearest")
    gy = convolve(img, ky, mode="nearest")

    ker = np.ones((m1, m1), dtype=np.float32)
    a = convolve(gx * gx, ker, mode="nearest")  # sum gx^2 over window
    d = convolve(gy * gy, ker, mode="nearest")  # sum gy^2 over window
    b = convolve(gx * gy, ker, mode="nearest")  # sum gx·gy over window

    a = a[:sH, :sW]
    d = d[:sH, :sW]
    b = b[:sH, :sW]

    tr = a + d
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4.0 * (b ** 2), 0.0))
    s1 = 0.5 * (tr + disc)
    return s1.reshape(-1).astype(np.float64)

def gamma_threshold(sigma, m1, delta=0.99):
    """
    τ = F^{-1}(δ; α=N/2, β=(2/N)σ^2·tr(DhᵀDh)).
    For an m1×m1 window with forward diff [-1,1]: tr(DhᵀDh) = 2*m1*(m1-1).
    """
    N = m1 * m1
    tr_Dh = 2.0 * m1 * (m1 - 1)
    alpha = N / 2.0
    beta = (2.0 / N) * (sigma ** 2) * tr_Dh
    tau = float(gamma_dist.ppf(delta, a=alpha, scale=beta))
    if not np.isfinite(tau) or tau <= 0:
        tau = (alpha * beta) * 3.0
    return tau

def covariances_from_patches(patches, ridge=1e-6, chunk_size=None, dtype=np.float64):
    s, m1, _, C = patches.shape
    if s == 0:
        raise RuntimeError("No patches selected (s=0). Relax selection or increase MIN_KEEP.")
    m2 = m1 * m1
    assert C == 3, "Expected 3 channels."

    X = patches.reshape(s, m2, C).astype(dtype, copy=False)
    I = np.eye(m2, dtype=dtype)
    covs = []

    if chunk_size is None:
        for j in range(C):
            Y = X[:, :, j].T                # (m2, s)
            mu = Y.mean(axis=1, keepdims=True)
            Z  = Y - mu
            Sigma = (Z @ Z.T) / float(s)    # (m2, m2)
            if ridge > 0:
                Sigma += ridge * I
            Sigma = 0.5 * (Sigma + Sigma.T)
            covs.append(Sigma)
        return covs

    chunk_size = max(int(chunk_size), 1)
    for j in range(C):
        mu = np.zeros((m2, 1), dtype=dtype)
        n = 0
        for start in range(0, s, chunk_size):
            end = min(start + chunk_size, s)
            block = X[start:end, :, j]      # (b, m2)
            mu += block.sum(axis=0, keepdims=True).T
            n  += (end - start)
        mu /= float(n)

        Sigma = np.zeros((m2, m2), dtype=dtype)
        for start in range(0, s, chunk_size):
            end = min(start + chunk_size, s)
            block = X[start:end, :, j]      # (b, m2)
            Zb = block - mu.T               # (b, m2)
            Sigma += Zb.T @ Zb
        Sigma /= float(n)

        if ridge > 0:
            Sigma += ridge * I
        Sigma = 0.5 * (Sigma + Sigma.T)
        covs.append(Sigma)
    return covs

def smallest_blockdiag_eigs_concat(covs, n):
    all_eigs = []
    for Σ in covs:
        w = np.linalg.eigvalsh(0.5 * (Σ + Σ.T))
        all_eigs.append(w)
    all_eigs = np.concatenate(all_eigs)

    all_eigs = np.nan_to_num(all_eigs, nan=np.inf, posinf=np.inf, neginf=0.0)
    all_eigs = np.maximum(all_eigs, 0.0)
    all_eigs.sort()

    lambdas = all_eigs[:n].astype(np.float64)
    sigma0_var = float(all_eigs[0])
    return lambdas, sigma0_var

def pick_theta_for_sigma0(sigma0_var):
    sigma0 = math.sqrt(max(sigma0_var, 0.0))
    nearest = TRAIN_SIGMAS[np.argmin(np.abs(TRAIN_SIGMAS - sigma0))]
    return THETA_TABLE[int(np.sqrt(ADD_NOISE_VARIANCE))]

def estimate_variance_from_eigs(lambdas, theta_vec):
    θ0 = theta_vec[0]
    θs = theta_vec[1:]
    L = lambdas[:len(θs)]
    if len(L) < len(θs):
        L = np.pad(L, (0, len(θs) - len(L)), mode='edge')
    return float(θ0 + np.dot(θs, L))

def weak_textured_mask_iterative(img_u8, patches_all, m1, initial_sigma,
                                 delta=0.99, max_iters=8, tol=0.25, min_keep=500):
    s1 = compute_s1_map(rgb_to_gray_u8(img_u8), m1)
    s_total = s1.size

    sigma = float(max(initial_sigma, 0.0))
    prev_sigma = None
    chosen_mask = np.zeros(s_total, dtype=bool)

    for it in range(max_iters):
        tau = gamma_threshold(sigma, m1, delta=delta)
        mask = (s1 < tau)

        if mask.sum() < min_keep:
            deficit = min_keep - mask.sum()
            if deficit > 0:
                delta_relaxed = min(0.999, delta + 0.005 * deficit / max(min_keep, 1))
                tau = gamma_threshold(sigma, m1, delta=delta_relaxed)
                mask = (s1 < tau)
        if mask.sum() < min_keep:
            k = min(min_keep, s_total)
            kth = np.partition(s1, k - 1)[k - 1]
            mask = (s1 <= kth)

        patches = patches_all[mask]
        covs = covariances_from_patches(patches, ridge=1e-6)
        lambdas, sigma0_var = smallest_blockdiag_eigs_concat(covs, N_EIGEN)
        theta = pick_theta_for_sigma0(sigma0_var)
        var_hat = max(estimate_variance_from_eigs(lambdas, theta), 0.0)
        sigma_new = math.sqrt(var_hat)

        if DEBUG_ITER_LOG:
            print(f"  iter {it+1}: δ={delta:.3f}, τ≈{tau:.2f}, selected={int(mask.sum())}, σ≈{sigma_new:.3f}")

        chosen_mask = mask
        if prev_sigma is not None and abs(sigma_new - prev_sigma) <= tol:
            sigma = sigma_new
            break
        prev_sigma = sigma
        sigma = sigma_new

    return chosen_mask, sigma

# --------------------------------
# Pipeline
# --------------------------------
def main():
    # 1) collect images
    exts = ("*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(IMG_DIR, e)))
    paths = sorted(paths)[:NUM_IMAGES]
    if len(paths) < NUM_IMAGES:
        raise RuntimeError(f"Found {len(paths)} images, need {NUM_IMAGES}")

    true_var = ADD_NOISE_VARIANCE
    true_sigma = math.sqrt(true_var)

    est_vars = []
    est_sigmas = []

    for p in paths:
        clean = load_rgb_uint8(p)
        noisy = add_gaussian_noise_uint8(clean, true_var)

        patches_all = sliding_3d_overlapping_patches(noisy, PATCH_M1)   # (s, m1, m1, 3)

        covs_all = covariances_from_patches(patches_all, ridge=1e-6)
        lambdas_all, sigma0_var_all = smallest_blockdiag_eigs_concat(covs_all, N_EIGEN)
        theta_all = pick_theta_for_sigma0(sigma0_var_all)
        var_hat_all = estimate_variance_from_eigs(lambdas_all, theta_all)
        sigma_init = math.sqrt(max(var_hat_all, 0.0))

        mask, _sigma_iter = weak_textured_mask_iterative(
            img_u8=noisy,
            patches_all=patches_all,
            m1=PATCH_M1,
            initial_sigma=sigma_init,
            delta=DELTA,
            max_iters=MAX_ITERS,
            tol=TOL_SIGMA,
            min_keep=MIN_KEEP
        )

        patches = patches_all[mask]
        covs = covariances_from_patches(patches, ridge=1e-6)
        lambdas, sigma0_var = smallest_blockdiag_eigs_concat(covs, N_EIGEN)
        theta = pick_theta_for_sigma0(sigma0_var)
        var_hat = max(estimate_variance_from_eigs(lambdas, theta), 0.0)
        sigma_hat = math.sqrt(var_hat)

        est_vars.append(var_hat)
        est_sigmas.append(sigma_hat)

        print(f"{os.path.basename(p):20s}  σ̂^2={var_hat:8.4f}  (σ̂={sigma_hat:6.3f};  σ0≈{math.sqrt(sigma0_var):.3f};  patches={patches.shape[0]})")

    est_vars = np.array(est_vars, dtype=np.float64)
    est_sigmas = np.array(est_sigmas, dtype=np.float64)

    bias_var = float(est_vars.mean() - true_var)
    bias_sigma = float(est_sigmas.mean() - true_sigma)
    mae_sigma = float(np.mean(np.abs(est_sigmas - true_sigma)))
    rmse_sigma = float(np.sqrt(np.mean((est_sigmas - true_sigma) ** 2)))

    print("\n=== Summary over {} images ===".format(NUM_IMAGES))
    print(f"True variance              : {true_var:.4f}   (σ = {true_sigma:.4f})")
    print(f"Mean estimated variance    : {est_vars.mean():.4f}")
    print(f"Mean estimated sigma       : {est_sigmas.mean():.4f}")
    print(f"Bias (variance)            : {bias_var:+.4f}  [E(σ̂^2) - σ^2]")
    print(f"Bias (sigma)               : {bias_sigma:+.4f} [E(σ̂)   - σ]")
    print(f"MAE (sigma)                : {mae_sigma:.4f}")
    print(f"RMSE (sigma)               : {rmse_sigma:.4f}")

if __name__ == "__main__":
    main()
