import numpy as np
from scipy.ndimage import convolve
from PIL import Image
from pathlib import Path
from scipy.stats import gamma as gamma_dist
from scipy.integrate import quad
from scipy.io import loadmat
import math
import os
import glob
import h5py

GAUS_DATA_PATH = Path("drift_detection/gaus_data.mat")
IMG_DIR = Path("D:/Datasets/tid2008/reference_images")
NUM_IMAGES = 25
ADD_NOISE_SIGMA = 30.0
PATCH_M1 = 8
DELTA = 0.8
MAX_ITERS = 3
DECIM = 0

def _load_gaus_y(path: str):
    try:
        data = loadmat(path)
        if "y" in data:
            y = np.asarray(data["y"]).squeeze()
            return y
        # sometimes MATLAB structs wrap fields
        for v in data.values():
            if isinstance(v, np.ndarray) and v.size > 0 and "y" in dir(v):
                y = np.asarray(v["y"]).squeeze()
                return y
        raise KeyError("Variable 'y' not found in MAT (pre-v7.3) file.")
    except NotImplementedError:
        pass

    with h5py.File(path, "r") as f:
        # common cases: 'y' at root, or nested like '/y', '/data/y'
        if "y" in f:
            ds = f["y"]
        else:
            # search for a dataset named 'y' anywhere
            ds = None
            def _visit(name, obj):
                nonlocal ds
                if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == "y":
                    ds = obj
            f.visititems(_visit)
            if ds is None:
                # if there’s only one dataset, use it
                dsets = []
                f.visititems(lambda n, o: dsets.append(o) if isinstance(o, h5py.Dataset) else None)
                if len(dsets) == 1:
                    ds = dsets[0]
                else:
                    raise KeyError("Variable 'y' not found in MAT v7.3 file.")
        y = np.array(ds[()]).squeeze()
        return y

def gaus_from_table(m: float, path: str = GAUS_DATA_PATH) -> float:
    y = _load_gaus_y(path).ravel()
    # MATLAB table assumes x spans [-10, 10] with step 0.001
    # length should be 20001; we reconstruct x implicitly via index
    diff = np.abs(y - float(m))
    idx = int(diff.argmin())
    # emulate MATLAB tolerance behavior: prefer |y-m| < 0.01 if any
    close = np.where(diff < 0.01)[0]
    if close.size > 0:
        idx = int(close[0])
    # x = -10 + 0.001 * idx
    return -10.0 + 0.001 * idx

def add_gaussian_noise_float(img_u8, sigma):
    x = img_u8.astype(np.float64)
    noise = np.random.randn(*x.shape) * float(sigma)
    return x + noise

def my_convmtx2(H, m, n):
    H = np.asarray(H, dtype=np.float64)
    s1, s2 = H.shape
    out_rows = (m - s1 + 1) * (n - s2 + 1)
    T = np.zeros((out_rows, m * n), dtype=np.float64)
    k = 0
    for i in range(m - s1 + 1):
        for j in range(n - s2 + 1):
            for p in range(s1):
                start = (i + p) * n + j
                end = start + s2
                T[k, start:end] = H[p, :]
            k += 1
    return T

def im2col_sliding(img2d, block_h, block_w):
    H, W = img2d.shape
    oh = H - block_h + 1
    ow = W - block_w + 1
    if oh <= 0 or ow <= 0:
        return np.zeros((block_h * block_w, 0), dtype=img2d.dtype)
    shape = (oh, ow, block_h, block_w)
    strides = (img2d.strides[0], img2d.strides[1], img2d.strides[0], img2d.strides[1])
    patches = np.lib.stride_tricks.as_strided(img2d, shape=shape, strides=strides)
    cols = patches.reshape(oh * ow, block_h * block_w).T
    return cols

def sort_lex_and_decimate(XtrX, decim):
    if XtrX.shape[1] == 0:
        return XtrX[:1, :], XtrX[1:, :], np.arange(0)
    order = np.lexsort(tuple(XtrX[i, :] for i in range(XtrX.shape[0] - 1, -1, -1)))
    XtrX_sorted = XtrX[:, order]
    if decim <= 0:
        keep_idx_sorted = np.arange(XtrX_sorted.shape[1])
    else:
        step = decim + 1
        pcount = int(np.floor(XtrX_sorted.shape[1] / step))
        keep_idx_sorted = (np.arange(1, pcount + 1) * step) - 1
    Xtr = XtrX_sorted[0:1, keep_idx_sorted]
    X = XtrX_sorted[1:, keep_idx_sorted]
    keep_idx_original = order[keep_idx_sorted]
    return Xtr, X, keep_idx_original

def my_gaussian(x):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)


def CumptLambda(delta, patch):
    r = patch * patch - 1
    sig = 10.0
    b = gamma_dist.ppf(delta, a=r / 2.0, scale=2.0 / r)
    t1 = 1.0 / math.gamma(r / 2.0)
    t2 = r / (2.0 * sig * sig)
    f = lambda x: (sig * sig * x) * t1 * ((sig * sig * x) ** (r / 2.0 - 1.0)) * (t2 ** (r / 2.0)) * np.exp(-t2 * sig * sig * x)
    g = lambda x: (sig * sig) * t1 * ((sig * sig * x) ** (r / 2.0 - 1.0)) * (t2 ** (r / 2.0)) * np.exp(-t2 * sig * sig * x)
    rho1, _ = quad(f, 0.0, b, limit=200)
    rho2, _ = quad(g, 0.0, b, limit=200)
    rho = rho1 / rho2
    lam = b / rho
    return lam, rho

def noise_lev_est(img, show, w, delta, decim, conf, itr):
    assert img.ndim == 3
    H, W, C = img.shape
    Yimg = img.astype(np.float64)
    if itr is None:
        itr = 3
    if conf is None:
        conf = 1 - 1e-6
    if decim is None:
        decim = 0
    patchsize = w
    kh = np.array([[-0.5, 0.0, 0.5]], dtype=np.float64)
    kv = kh.T
    imgh = np.empty((H, W - 2, C), dtype=Yimg.dtype)
    imgv = np.empty((H - 2, W, C), dtype=Yimg.dtype)
    for c in range(C):
        th = convolve(Yimg[:, :, c], kh, mode='nearest')
        th = th[:, 1:-1]
        imgh[:, :, c] = th * th
        tv = convolve(Yimg[:, :, c], kv, mode='nearest')
        tv = tv[1:-1, :]
        imgv[:, :, c] = tv * tv
    Dh = my_convmtx2(kh, patchsize, patchsize)
    Dv = my_convmtx2(kv, patchsize, patchsize)
    DD = Dh.T @ Dh + Dv.T @ Dv
    rDD = np.linalg.matrix_rank(DD)
    Dtr = np.trace(DD)
    tau0 = gamma_dist.ppf(conf, a=rDD / 2.0, scale=2.0 * Dtr / rDD)
    lam, rho = CumptLambda(delta, patchsize)
    nlevel = np.zeros((C,), dtype=np.float64)
    th_out = np.zeros((C,), dtype=np.float64)
    num = np.zeros((C,), dtype=np.int64)
    for cha in range(C):
        X = im2col_sliding(Yimg[:, :, cha], patchsize, patchsize)
        X0 = X.copy()
        Xh = im2col_sliding(imgh[:, :, cha], patchsize, patchsize - 2)
        Xv = im2col_sliding(imgv[:, :, cha], patchsize - 2, patchsize)
        if not (X.shape[1] == Xh.shape[1] == Xv.shape[1]):
            raise ValueError("Patch counts do not align.")
        Xtr = np.sum(np.vstack([Xh, Xv]), axis=0, keepdims=True)
        if decim > 0:
            XtrX = np.vstack([Xtr, X])
            Xtr, X, _ = sort_lex_and_decimate(XtrX, decim)
        if X.shape[1] < X.shape[0]:
            sig2 = 0.0
        else:
            cov = (X @ X.T) / (X.shape[1] - 1)
            d = np.linalg.eigvalsh(cov)
            sig2 = float(d[0])
        p1 = np.arange(X0.shape[1])
        tau = sig2 * tau0
        d1 = 0; d2 = 0; t1 = 0; t2 = 0
        for _i in range(2, (itr if itr is not None else 3) + 1):
            mask = (Xtr.flatten() < tau)
            Xtr = Xtr[:, mask]
            X = X[:, mask]
            p1 = p1[mask]
            if X.shape[1] < X.shape[0]:
                break
            cov = (X @ X.T) / (X.shape[1] - 1)
            _ = np.linalg.eigvalsh(cov)
            X00 = X
            SigSet = np.std(X00, axis=0, ddof=0)
            Idx = np.argsort(SigSet)
            if Idx.size >= 10:
                Idx = Idx[:10]
            sigw = float(np.mean(SigSet[Idx])) if Idx.size > 0 else float(np.mean(SigSet))
            it_star = sigw
            diff0 = it_star
            it_end = float(np.mean(SigSet))
            max_step = 100
            it_step = (it_end - it_star) / max_step if max_step > 0 else 0.0
            while diff0 > 0:
                sigw = sigw + it_step
                idx = np.where(SigSet <= (lam * sigw))[0]
                num_v = idx.size
                if num_v == 0:
                    diff0 = -1.0
                    break
                diff0 = float(np.sum(SigSet[idx] - sigw) / num_v)
            if diff0 == 0 or idx.size == 0:
                break
            XX = X00[:, idx]
            LP = idx.size
            if XX.shape[1] < 2:
                break
            n = patchsize * patchsize
            alpha = 0.375
            s = float(LP)
            x_val = (rho - 1.0) * np.sqrt(s / 2.0)
            Phi_int, _ = quad(my_gaussian, -100.0, x_val, limit=200)
            i0 = n - alpha + 1.0 - Phi_int * (n - 2.0 * alpha + 1.0)
            i0 = int(np.floor(i0))
            i0 = max(1, min(n, i0))
            z1 = gaus_from_table((1.0 - alpha) / (n - 2.0 * alpha + 1.0))
            d1 = abs(-z1 * np.sqrt(2.0 / s))
            z2 = gaus_from_table((n - alpha + 1.0 - i0) / (n - 2.0 * alpha + 1.0))
            d2 = abs(((1.0 - rho) + z2 * np.sqrt(2.0 / s)) / rho)
            c_mat = (XX @ XX.T) / (XX.shape[1] - 1)
            cd = np.linalg.eigvalsh(c_mat)[::-1]
            idx0 = min(len(cd) - 1, max(0, i0 - 1))
            t1 = cd[idx0] / rho
            t2 = cd[-1]
            t22 = (d2 * t2 + t1 * d1) / (d1 + d2 + 1e-12)
            tau = t22 * tau
        nlevel[cha] = np.sqrt((d2 * t2 + t1 * d1) / (d1 + d2 + 1e-12))
        th_out[cha] = float(tau)
        num[cha] = int(X.shape[1])
    return nlevel, th_out, num

def main():
    exts = ("*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(IMG_DIR, e)))
    paths = sorted(paths)[:NUM_IMAGES]
    if len(paths) < NUM_IMAGES:
        raise RuntimeError(f"Found {len(paths)} images, need {NUM_IMAGES}")

    est_sigmas = []
    biases = []
    for p in paths:
        # true_sigma = np.random.uniform(0.0, ADD_NOISE_SIGMA)
        true_sigma = ADD_NOISE_SIGMA
        clean = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        noisy = add_gaussian_noise_float(clean, true_sigma)
        noisy_img = noisy.clip(0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img.astype(np.uint8))
        noisy_img.save(f"{p}_{ADD_NOISE_SIGMA}_noisy.bmp")
        nlevel, th, num = noise_lev_est(noisy, show=False, w=PATCH_M1, delta=DELTA, decim=DECIM, conf=1 - 1e-6, itr=MAX_ITERS)
        est_sigma = nlevel.mean()
        est_sigmas.append(est_sigma)
        biases.append(np.abs(est_sigma - true_sigma))
        print(f"{os.path.basename(p):20s}  σ={true_sigma:6.3f}  (σ̂={est_sigma:6.3f}; bias={np.abs(est_sigma - true_sigma):6.3f})")
    est_sigmas = np.array(est_sigmas, dtype=np.float64)
    biases = np.array(biases, dtype=np.float64)
    print(f"\nbias={biases.mean():.4f}")

if __name__ == "__main__":
    main()
