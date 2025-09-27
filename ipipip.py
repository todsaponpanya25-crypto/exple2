# fundus_enhancement_app.py
# Retinal fundus enhancement with paper-style visualization
# Pipeline: TV decomposition -> Visual adaptation (Naka–Rushton) -> Weighted fusion

import io
import math
from typing import Tuple

# ---- Safe OpenCV import with helpful message ----
try:
    import cv2
except ModuleNotFoundError:
    import streamlit as st
    st.error(
        "OpenCV (cv2) is not installed. Install it with:\n\n"
        "`pip install opencv-python-headless`\n\n"
        "If you're using conda: `conda install -c conda-forge opencv`"
    )
    st.stop()

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ===========================
# ====== UI CONFIG ==========
# ===========================
st.set_page_config(page_title="Retinal Fundus Enhancement — Paper-Style", page_icon="🩺", layout="wide")
st.title("🩺 Retinal Fundus Enhancement — Paper-Style Visualization")
st.caption("Upload a fundus image, or use the sample. Defaults match the paper; visuals explain each processing stage.")

# ===========================
# ====== UTILITIES ==========
# ===========================
def to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    m, M = float(img.min()), float(img.max())
    if M <= m:
        return np.zeros_like(img, dtype=np.float32)
    return (img - m) / (M - m)

def to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)

def rgb_to_hsv(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_uint8(rgb01), cv2.COLOR_RGB2HSV).astype(np.float32)

def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return to_float01(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

def gaussian_from_sigma(sigma: float):
    k = max(3, int(6 * sigma + 1))
    if k % 2 == 0:
        k += 1
    return (k, k), sigma

def imshow_heat(ax, img01, title="", cmap="magma", vmin=None, vmax=None):
    ax.imshow(img01, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

def imshow_rgb(ax, img01, title=""):
    ax.imshow(np.clip(img01, 0, 1))
    ax.set_title(title, fontsize=11)
    ax.axis("off")

def hist_plot(ax, data01, title="", bins=256):
    ax.hist(data01.ravel(), bins=bins, range=(0, 1), alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")

def profile_line(img01_gray, r0, r1, c0, c1, num=512):
    # Bresenham-like sampling for a straight line profile
    ys = np.linspace(r0, r1, num)
    xs = np.linspace(c0, c1, num)
    coords = np.stack([ys, xs], axis=1)
    H, W = img01_gray.shape[:2]
    vals = []
    for y, x in coords:
        yi = int(np.clip(round(y), 0, H-1))
        xi = int(np.clip(round(x), 0, W-1))
        vals.append(img01_gray[yi, xi])
    return np.array(vals), coords

# ===========================
# ====== SAMPLE IMAGE =======
# ===========================
def demo_fundus() -> np.ndarray:
    """Synthetic green-dominant disc + vessel strokes, shape-safe."""
    H, W = 640, 640
    y, x = np.indices((H, W))
    cx, cy, r = W // 2, H // 2, int(0.8 * H / 2)
    img = np.zeros((H, W, 3), np.float32)

    # Disc with smooth falloff in G
    dist2 = (y - cy) ** 2 + (x - cx) ** 2
    mask = dist2 <= r * r
    falloff = np.clip(1.0 - dist2 / float(r * r), 0.0, 1.0)
    channel_g = 0.15 + 0.5 * falloff
    channel_g[~mask] = 0.0
    img[..., 1] = channel_g

    # Optic-disc-like region
    cv2.circle(img, (cx - 80, cy - 60), 42, (0.3, 0.25, 0.2), -1)

    # Vessel strokes
    for xoff in range(-240, 241, 40):
        cv2.line(img, (cx + xoff, cy - 220), (cx + xoff // 2, cy + 220), (0.1, 0.5, 0.1), 1)

    return np.clip(img, 0, 1)

# ==================================
# == Chambolle ROF TV (grayscale) ===
# ==================================
def tv_denoise_gray(u0: np.ndarray, weight: float, tau: float = 0.125, iters: int = 50) -> np.ndarray:
    u = u0.copy()
    px = np.zeros_like(u0)
    py = np.zeros_like(u0)
    lam = max(1e-8, float(weight))
    for _ in range(iters):
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        px_new = px + (tau / lam) * ux
        py_new = py + (tau / lam) * uy
        norm = np.maximum(1.0, np.sqrt(px_new * px_new + py_new * py_new))
        px, py = px_new / norm, py_new / norm
        div = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
        u = (u0 + lam * div)
    return np.clip(u, 0.0, 1.0)

def tv_decompose_color(img01: np.ndarray, lam: Tuple[float, float, float], iters: int = 50):
    base = np.zeros_like(img01)
    for c in range(3):
        base[..., c] = tv_denoise_gray(img01[..., c], lam[c], iters=iters)
    detail = img01 - base
    return base, detail

# ==================================
# === Global noise estimation λ1 ====
# ==================================
NS = np.array([[1, -2, 1],
               [-2, 4, -2],
               [1, -2, 1]], np.float32)

def lambda1_global_noise(img01: np.ndarray) -> Tuple[float, float, float]:
    H, W = img01.shape[:2]
    lams = []
    for c in range(3):
        ch = img01[..., c]
        conv = cv2.filter2D(ch, -1, NS, borderType=cv2.BORDER_REFLECT)
        s = float(np.sum(np.abs(conv)))
        lam = math.sqrt(math.pi / 2.0) / (6.0 * max(W - 2, 1) * max(H - 2, 1)) * s
        lams.append(lam + 1e-6)
    return tuple(lams)

# ==================================
# === Visual adaptation (HSV.V) ====
# ==================================
def visual_adaptation_on_base(base_rgb01: np.ndarray, n: float = 1.0):
    hsv = rgb_to_hsv(base_rgb01)
    V = to_float01(hsv[..., 2])
    Mg = float(np.mean(V))
    Sg = float(np.std(V))
    sigma_g = Mg / (1.0 + math.exp(Sg))  # paper's σ_g
    outV = (V ** n) / ((V ** n) + (sigma_g ** n + 1e-12))
    hsv[..., 2] = np.clip(outV * 255.0, 0, 255)
    return hsv_to_rgb(hsv), V, outV, sigma_g

# ==================================
# ===== Weighted fusion (ω_c) ======
# ==================================
def fuse_with_weights(enh_base_rgb01: np.ndarray, detail_rgb01: np.ndarray,
                      alpha_r: float, alpha_g: float, alpha_b: float, gauss_sigma: float):
    mag = np.abs(detail_rgb01)
    (kx, ky), sig = gaussian_from_sigma(gauss_sigma)
    w = np.zeros_like(detail_rgb01)
    alphas = [alpha_r, alpha_g, alpha_b]
    for c, a in enumerate(alphas):
        blur = cv2.GaussianBlur(mag[..., c], (kx, ky), sig, borderType=cv2.BORDER_REFLECT)
        w[..., c] = a * blur
    out = np.clip(enh_base_rgb01 + w * detail_rgb01, 0.0, 1.0)
    return out, w

# ==================================
# ============== UI ================
# ==================================
with st.sidebar:
    st.header("1) Upload fundus image")
    use_sample_until_upload = st.toggle(
        "Use sample image until upload",
        value=True,
        help="If off, the app will wait for your upload and stop."
    )
    f = st.file_uploader("PNG/JPG", type=["png", "jpg", "jpeg"])

    st.header("2) Defaults")
    use_paper_defaults = st.checkbox(
        "Use paper defaults",
        True,
        help="λ₂=0.3, α_R=α_G=600, α_B=0, Gaussian σ=10, n=1.0"
    )

    st.header("3) Decomposition")
    iters = st.slider("TV iterations", 20, 200, 60, 5)
    lam2 = 0.3 if use_paper_defaults else st.slider("λ₂ (base/detail TV)", 0.05, 1.0, 0.3, 0.01)

    st.header("4) Visual adaptation")
    n_naka = 1.0 if use_paper_defaults else st.slider("Naka–Rushton n", 0.5, 3.0, 1.0, 0.1)

    st.header("5) Fusion weights")
    alpha_r = 600.0 if use_paper_defaults else st.number_input("α_R (emphasize veins)", 0.0, 2000.0, 600.0, 10.0)
    alpha_g = 600.0 if use_paper_defaults else st.number_input("α_G (emphasize arteries)", 0.0, 2000.0, 600.0, 10.0)
    alpha_b = 0.0 if use_paper_defaults else st.number_input("α_B (blue/artifacts)", 0.0, 2000.0, 0.0, 10.0)
    gauss_sigma = 10.0 if use_paper_defaults else st.slider("Gaussian σ (for ω)", 1.0, 20.0, 10.0, 0.5)

# ==================================
# ========= Load the image ==========
# ==================================
if f is not None:
    img = Image.open(f).convert("RGB")
    rgb01 = to_float01(np.array(img))
else:
    if use_sample_until_upload:
        rgb01 = demo_fundus()
        st.info("No image uploaded — using the built-in sample fundus. Upload a real fundus image to process it.")
    else:
        st.warning("Please upload a retinal fundus image to continue.")
        st.stop()

st.caption(
    "Pipeline: (1) TV decomposition (λ₁ noise→structure) → (2) TV (λ₂ base→detail) → "
    "(3) Naka–Rushton visual adaptation on luminance → (4) fuse enhanced base + weighted detail (discard noise)."
)

# ==================================
# ========== Processing =============
# ==================================
# STEP 1: noise vs structure using λ₁
lam1_r, lam1_g, lam1_b = lambda1_global_noise(rgb01)
structure, noise = tv_decompose_color(rgb01, (lam1_r, lam1_g, lam1_b), iters=iters)

# STEP 2: base vs detail from structure (λ₂)
base, detail = tv_decompose_color(structure, (lam2, lam2, lam2), iters=iters)

# STEP 3: visual adaptation on base (HSV.V using Naka–Rushton)
enh_base, V_before, V_after, sigma_g = visual_adaptation_on_base(base, n=n_naka)

# STEP 4: fusion (discard noise layer) + keep weight maps
out, weights = fuse_with_weights(enh_base, detail, alpha_r, alpha_g, alpha_b, gauss_sigma)

# ==================================
# ======== TABS (Paper-style) =======
# ==================================
tab_overview, tab_decomp, tab_adapt, tab_fuse, tab_profiles = st.tabs(
    ["Overview", "Decomposition", "Visual Adaptation", "Fusion", "Line Profiles"]
)

# ---------- OVERVIEW ----------
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(to_uint8(rgb01), channels="RGB", use_container_width=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        hist_plot(ax, to_float01(cv2.cvtColor(to_uint8(rgb01), cv2.COLOR_RGB2HSV))[...,2]/255.0, "Histogram (Original luminance V)")
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.subheader("Enhanced")
        st.image(to_uint8(out), channels="RGB", use_container_width=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        hist_plot(ax, V_after, "Histogram (Enhanced luminance V)")
        st.pyplot(fig, use_container_width=True)

# ---------- DECOMPOSITION ----------
with tab_decomp:
    st.markdown("**Two-stage TV decomposition**: First remove noise (λ₁), then split base/detail (λ₂).")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    imshow_rgb(axes[0,0], rgb01, "Input RGB")
    imshow_rgb(axes[0,1], structure, "Structure (after λ₁)")
    # visualize noise magnitude
    noise_mag = np.mean(np.abs(noise), axis=2)
    imshow_heat(axes[0,2], noise_mag, "Noise magnitude ‖N‖", cmap="magma")

    imshow_rgb(axes[1,0], base, "Base (after λ₂)")
    # detail magnitude
    detail_mag = np.mean(np.abs(detail), axis=2)
    imshow_heat(axes[1,1], detail_mag, "Detail magnitude ‖D‖", cmap="magma")
    imshow_rgb(axes[1,2], enh_base, "Enhanced base (after visual adaptation)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ---------- VISUAL ADAPTATION ----------
with tab_adapt:
    st.markdown("**Naka–Rushton adaptation** on luminance V of HSV: "
                r"$V' = \frac{V^n}{V^n + \sigma_g^n}$ "
                f"with **n={n_naka:.2f}** and **σ_g={sigma_g:.4f}**.")

    # Plot the response curve
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    v = np.linspace(0, 1, 512)
    resp = (v**n_naka) / (v**n_naka + (sigma_g**n_naka + 1e-12))
    ax.plot(v, resp, lw=2)
    ax.set_title("Naka–Rushton Response Curve")
    ax.set_xlabel("Input luminance V")
    ax.set_ylabel("Adapted luminance V'")
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # Before/After luminance heatmaps and histograms
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        imshow_heat(ax, V_before, "Luminance V (before)", cmap="magma")
        st.pyplot(fig, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        hist_plot(ax, V_before, "Histogram V (before)")
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        imshow_heat(ax, V_after, "Luminance V' (after)", cmap="magma")
        st.pyplot(fig, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        hist_plot(ax, V_after, "Histogram V' (after)")
        st.pyplot(fig, use_container_width=True)

# ---------- FUSION ----------
with tab_fuse:
    st.markdown("**Weighted fusion**: "
                r"$I_{out} = I_{enh\_base} + \omega_c \cdot D_c$, where "
                r"$\omega_c = \alpha_c (|D_c| * \mathcal{N}_\sigma)$.")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for c, name in enumerate(["R", "G", "B"]):
            imshow_heat(axes[0,c], np.abs(detail[..., c]), f"|Detail| channel {name}", cmap="magma")
        for c, name in enumerate(["R", "G", "B"]):
            imshow_heat(axes[1,c], weights[..., c], f"Weight ω_{name}", cmap="viridis")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.image(to_uint8(enh_base), caption="Enhanced Base (pre-fusion)", use_container_width=True)
        st.image(to_uint8(out), caption="Final Output (post-fusion)", use_container_width=True)

# ---------- LINE PROFILES ----------
with tab_profiles:
    st.markdown("**Contrast along a line**: compares grayscale profiles before/after to illustrate vessel enhancement.")
    # Choose a central horizontal line across the optic region
    H, W = rgb01.shape[:2]
    r0 = r1 = H // 2
    c0, c1 = W // 6, 5 * W // 6

    # Use green channel (clinically informative) for profiles
    g_before = rgb01[..., 1]
    g_after  = out[..., 1]
    prof_before, coords = profile_line(g_before, r0, r1, c0, c1, num=800)
    prof_after, _ = profile_line(g_after,  r0, r1, c0, c1, num=800)

    # Show the sampling line
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(rgb01)
    ax[0].plot([c0, c1], [r0, r1], lw=2)
    ax[0].set_title("Sampling line on Original")
    ax[0].axis("off")
    ax[1].imshow(out)
    ax[1].plot([c0, c1], [r0, r1], lw=2)
    ax[1].set_title("Sampling line on Enhanced")
    ax[1].axis("off")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Plot profiles
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(np.linspace(0, 1, prof_before.size), prof_before, label="Before", lw=1.8)
    ax.plot(np.linspace(0, 1, prof_after.size),  prof_after,  label="After", lw=1.8)
    ax.set_title("Intensity Profile Along the Line (G channel)")
    ax.set_xlabel("Normalized distance")
    ax.set_ylabel("Intensity")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# ==================================
# =========== Download =============
# ==================================
buf = io.BytesIO()
Image.fromarray(to_uint8(out)).save(buf, format="PNG")
st.download_button("⬇️ Download enhanced PNG", data=buf.getvalue(),
                   file_name="fundus_enhanced.png", mime="image/png")

# ==================================
# ======= Deployment notes =========
# ==================================
st.caption("Tip: For servers or Streamlit Cloud, depend on `opencv-python-headless` to avoid GUI deps.")
