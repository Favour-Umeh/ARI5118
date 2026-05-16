"""
Interactive Feature Pyramid Network (FPN) Simulator
Streamlit application for exploring multi-scale feature extraction,
top-down pathways, and lateral connections in FPN architectures.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import streamlit as st
from typing import Dict, List, Tuple, Optional
import time

# -----------------------------------------------------------------------------
# Helper Functions for Image Processing and Convolutions
# -----------------------------------------------------------------------------

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian kernel for convolution."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) *
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)


def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform 2D convolution with zero padding.
    Handles grayscale images (2D arrays) only.
    """
    if image.ndim == 3:
        image = np.mean(image, axis=2)  # convert to grayscale if RGB
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)

    return output


def downsample(feature_map: np.ndarray, stride: int) -> np.ndarray:
    """Downsample a feature map using average pooling with the given stride."""
    h, w = feature_map.shape
    new_h, new_w = h // stride, w // stride
    downsampled = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            block = feature_map[i * stride:(i + 1) * stride, j * stride:(j + 1) * stride]
            downsampled[i, j] = np.mean(block)
    return downsampled


def upsample_nearest(feature_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Upsample a feature map using nearest-neighbor interpolation."""
    h, w = feature_map.shape
    target_h, target_w = target_shape
    row_ratio = target_h / h
    col_ratio = target_w / w
    upsampled = np.zeros((target_h, target_w))
    for i in range(target_h):
        for j in range(target_w):
            orig_i = int(i / row_ratio)
            orig_j = int(j / col_ratio)
            upsampled[i, j] = feature_map[orig_i, orig_j]
    return upsampled


def upsample_bilinear(feature_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Upsample a feature map using bilinear interpolation."""
    h, w = feature_map.shape
    target_h, target_w = target_shape
    upsampled = np.zeros((target_h, target_w))
    scale_h = (h - 1) / (target_h - 1) if target_h > 1 else 0
    scale_w = (w - 1) / (target_w - 1) if target_w > 1 else 0

    for i in range(target_h):
        for j in range(target_w):
            src_i = i * scale_h
            src_j = j * scale_w

            i0 = int(np.floor(src_i))
            i1 = min(i0 + 1, h - 1)
            j0 = int(np.floor(src_j))
            j1 = min(j0 + 1, w - 1)

            di = src_i - i0
            dj = src_j - j0

            val = (feature_map[i0, j0] * (1 - di) * (1 - dj) +
                   feature_map[i1, j0] * di * (1 - dj) +
                   feature_map[i0, j1] * (1 - di) * dj +
                   feature_map[i1, j1] * di * dj)
            upsampled[i, j] = val
    return upsampled


def normalize_heatmap(map_array: np.ndarray) -> np.ndarray:
    """Normalize a feature map to [0, 1] for visualization."""
    min_val = np.min(map_array)
    max_val = np.max(map_array)
    if max_val - min_val < 1e-8:
        return np.zeros_like(map_array)
    return (map_array - min_val) / (max_val - min_val)


# -----------------------------------------------------------------------------
# Synthetic Image Generation (with multi-scale objects)
# -----------------------------------------------------------------------------

def generate_synthetic_image(size: int = 256) -> np.ndarray:
    """
    Generate a synthetic grayscale image containing objects of different scales:
    - Small circles (radius 3-5)
    - Medium squares (size 12-16)
    - Large ellipses (major axis ~30-40)
    This mimics realistic multi-scale detection scenarios.
    """
    img = Image.new('L', (size, size), color=80)
    draw = ImageDraw.Draw(img)

    # Add random noise background
    np_img = np.array(img, dtype=np.float32)
    np_img += np.random.normal(0, 5, (size, size))

    # Small objects (hard to detect with coarse features)
    for _ in range(6):
        x = np.random.randint(20, size - 20)
        y = np.random.randint(20, size - 20)
        r = np.random.randint(3, 6)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=200)

    # Medium objects
    for _ in range(4):
        x = np.random.randint(30, size - 30)
        y = np.random.randint(30, size - 30)
        s = np.random.randint(12, 17)
        draw.rectangle([x - s, y - s, x + s, y + s], fill=180)

    # Large objects
    for _ in range(2):
        x = np.random.randint(50, size - 50)
        y = np.random.randint(50, size - 50)
        rx = np.random.randint(25, 40)
        ry = np.random.randint(25, 40)
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=160, outline=200)

    # Add some structure (stripes) to give semantic meaning
    for i in range(0, size, 20):
        draw.line([(i, 0), (i, size)], fill=120, width=1)
        draw.line([(0, i), (size, i)], fill=120, width=1)

    return np.array(img, dtype=np.float32) / 255.0


# -----------------------------------------------------------------------------
# FPN Simulation Core
# -----------------------------------------------------------------------------

class FPNSimulator:
    """
    Simulates a Feature Pyramid Network using synthetic feature extraction.
    For each pyramid level (P2-P5 with strides 4,8,16,32), we:
    - Compute bottom-up features C_i using scale-specific kernels
    - Apply top-down upsampling and lateral connections to produce P_i
    """

    def __init__(self, image: np.ndarray, lateral_weight: float = 1.0,
                 upsampling_method: str = 'bilinear'):
        """
        Args:
            image: Grayscale image normalized to [0,1], shape (H, W)
            lateral_weight: Strength of lateral connections (0.0 to 2.0)
            upsampling_method: 'nearest' or 'bilinear'
        """
        self.image = image
        self.lateral_weight = lateral_weight
        self.upsampling_method = upsampling_method
        self.strides = {2: 4, 3: 8, 4: 16, 5: 32}  # level -> stride
        self.shapes = {}
        self.bottom_up = {}   # C2, C3, C4, C5
        self.top_down = {}    # T2, T3, T4, T5 (before fusion)
        self.pyramid = {}     # P2, P3, P4, P5 (final outputs)

        self._compute_bottom_up()
        self._compute_fpn()

    def _compute_bottom_up(self):
        """
        Compute bottom-up feature maps (simulating a CNN backbone).
        For each level, we apply a convolution with a kernel sensitive to
        a specific scale, then downsample to the target stride resolution.
        """
        for level, stride in self.strides.items():
            # Kernel size and sigma proportional to stride to simulate receptive field
            kernel_size = min(stride * 2 + 1, 15)  # limit kernel size
            sigma = stride / 4.0
            kernel = gaussian_kernel(kernel_size, sigma)

            # Apply convolution to extract features at this scale
            response = conv2d(self.image, kernel)

            # Downsample to target stride
            feat_map = downsample(response, stride)

            # Simulate channel depth by creating 3 "pseudo-channels"
            # For visualization we keep a 2D representation (mean across pseudo-channels)
            # Actually, we treat this as a single representative activation map.
            self.bottom_up[level] = feat_map
            self.shapes[level] = feat_map.shape

    def _upsample(self, feat_map: np.ndarray, target_level: int) -> np.ndarray:
        """Upsample a feature map to match the spatial size of target_level's bottom-up map."""
        target_shape = self.shapes[target_level]
        if self.upsampling_method == 'nearest':
            return upsample_nearest(feat_map, target_shape)
        else:
            return upsample_bilinear(feat_map, target_shape)

    def _compute_fpn(self):
        """
        Build the FPN top-down pathway with lateral connections:
        Start from C5 (coarsest, richest semantics), iteratively upsample
        and add lateral connections from corresponding bottom-up maps.
        """
        # P5 is just C5 after a 3x3 convolution (simulated as identity + slight smoothing)
        self.pyramid[5] = self.bottom_up[5]

        # Top-down: from level 5 down to level 2
        for level in range(5, 2, -1):
            # Upsample the higher-level pyramid feature
            upsampled = self._upsample(self.pyramid[level], level - 1)

            # Lateral connection: bottom-up feature at this level (C_{level-1}) projected
            # We simulate the 1x1 conv by a simple scaling (channel reduction is abstracted)
            lateral = self.bottom_up[level - 1] * self.lateral_weight

            # Element-wise addition (top-down + lateral)
            # Ensure same shape
            if upsampled.shape != lateral.shape:
                # Should not happen if strides are powers of two, but just in case
                lateral = self._match_shape(lateral, upsampled.shape)

            merged = upsampled + lateral

            # Apply a 3x3 convolution to reduce aliasing (simulated as light smoothing)
            smooth_kernel = gaussian_kernel(3, 0.5)
            merged = conv2d(merged, smooth_kernel)
            # Trim to exact shape (conv2d adds padding)
            merged = merged[:self.shapes[level - 1][0], :self.shapes[level - 1][1]]

            self.pyramid[level - 1] = merged

    def _match_shape(self, arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize array to target_shape using cropping/padding if needed."""
        h, w = arr.shape
        th, tw = target_shape
        if h > th:
            arr = arr[:th, :]
        elif h < th:
            pad_h = th - h
            arr = np.pad(arr, ((0, pad_h), (0, 0)), mode='edge')
        if w > tw:
            arr = arr[:, :tw]
        elif w < tw:
            pad_w = tw - w
            arr = np.pad(arr, ((0, 0), (0, pad_w)), mode='edge')
        return arr

    def get_bottom_up(self) -> Dict[int, np.ndarray]:
        """Return dictionary of bottom-up feature maps (C2-C5)."""
        return self.bottom_up

    def get_pyramid(self) -> Dict[int, np.ndarray]:
        """Return dictionary of final FPN output maps (P2-P5)."""
        return self.pyramid

    def compute_detection_scores(self, ground_truth_objects: Optional[List[Tuple]] = None) -> Dict[str, float]:
        """
        Simulate detection performance: computes average activation in object regions.
        Returns a dictionary with scores for small, medium, large objects.
        This illustrates how FPN improves multi-scale detection.
        """
        # For demonstration, we define three size categories based on feature map resolution:
        # Small -> P2, Medium -> P3, Large -> P4/P5
        scores = {'small': 0.0, 'medium': 0.0, 'large': 0.0}
        if not ground_truth_objects:
            # Use synthetic ground truth based on image content: we approximate by measuring
            # activation variance in P2 vs P5. This is illustrative.
            p2 = self.pyramid.get(2)
            p5 = self.pyramid.get(5)
            if p2 is not None:
                scores['small'] = np.mean(p2) * 10  # arbitrary scaling for visualization
                scores['medium'] = np.mean(self.pyramid.get(3, p2)) * 10
                scores['large'] = np.mean(p5) * 10 if p5 is not None else 0
        return scores


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def plot_feature_grid(feature_maps: Dict[int, np.ndarray], title: str,
                      colormap: str = 'viridis', figsize=(12, 3)):
    """Plot a row of feature maps for levels 2,3,4,5."""
    levels = sorted(feature_maps.keys())
    fig, axes = plt.subplots(1, len(levels), figsize=figsize)
    if len(levels) == 1:
        axes = [axes]
    for ax, level in zip(axes, levels):
        feat = feature_maps[level]
        norm_feat = normalize_heatmap(feat)
        im = ax.imshow(norm_feat, cmap=colormap, interpolation='bilinear')
        ax.set_title(f'P{level}\n{feat.shape[0]}x{feat.shape[1]} (stride={4*2**(level-2)})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    return fig


def plot_comparison_side_by_side(fpn_left: FPNSimulator, fpn_right: FPNSimulator,
                                 param_names: Tuple[str, str], colormap='plasma'):
    """Side-by-side comparison of two FPN configurations."""
    left_pyr = fpn_left.get_pyramid()
    right_pyr = fpn_right.get_pyramid()

    levels = [2, 3, 4, 5]
    fig, axes = plt.subplots(2, len(levels), figsize=(14, 6))

    for idx, level in enumerate(levels):
        # Left column
        ax_left = axes[0, idx]
        feat_left = left_pyr[level]
        ax_left.imshow(normalize_heatmap(feat_left), cmap=colormap, interpolation='bilinear')
        ax_left.set_title(f'{param_names[0]} - P{level}\n{feat_left.shape}')
        ax_left.axis('off')

        # Right column
        ax_right = axes[1, idx]
        feat_right = right_pyr[level]
        ax_right.imshow(normalize_heatmap(feat_right), cmap=colormap, interpolation='bilinear')
        ax_right.set_title(f'{param_names[1]} - P{level}\n{feat_right.shape}')
        ax_right.axis('off')

    fig.suptitle('FPN Output Feature Maps - Side by Side Comparison', fontsize=16)
    plt.tight_layout()
    return fig


def plot_detection_bar_chart(scores_left: Dict, scores_right: Optional[Dict] = None,
                             labels=('Config A', 'Config B')):
    """Bar chart comparing detection scores for object sizes."""
    categories = list(scores_left.keys())
    left_vals = [scores_left[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, left_vals, width, label=labels[0], color='skyblue')
    if scores_right:
        right_vals = [scores_right[cat] for cat in categories]
        bars2 = ax.bar(x + width/2, right_vals, width, label=labels[1], color='lightcoral')

    ax.set_ylabel('Activation Score (simulated)')
    ax.set_title('Multi-Scale Detection Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    return fig


# -----------------------------------------------------------------------------
# Streamlit Application
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="FPN Interactive Simulator", layout="wide")
    st.title("📐 Feature Pyramid Network (FPN) Simulator")
    st.markdown("""
    **Explore how Feature Pyramid Networks enable robust multi-scale object detection.**
    Adjust parameters to see real-time changes in feature maps and detection performance.
    """)

    # Sidebar: Global controls and mode selection
    st.sidebar.header("⚙️ Global Settings")
    mode = st.sidebar.radio("Visualization Mode", ["Single Configuration", "Side-by-Side Comparison"])

    # Image source
    img_source = st.sidebar.radio("Image Source", ["Synthetic (default)", "Upload your own"])
    image = None
    if img_source == "Synthetic (default)":
        image = generate_synthetic_image(256)
    else:
        uploaded = st.sidebar.file_uploader("Upload an image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
        if uploaded:
            pil_img = Image.open(uploaded).convert('L')
            image = np.array(pil_img, dtype=np.float32) / 255.0
            # Resize to reasonable size (256x256) to keep computation fast
            if image.shape[0] != 256 or image.shape[1] != 256:
                pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
                image = np.array(pil_img, dtype=np.float32) / 255.0

    if image is None:
        st.warning("Using default synthetic image. Upload an image for custom experiments.")
        image = generate_synthetic_image(256)

    # Display original image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Input Image (grayscale)", clamp=True, width=300)

    # Parameter sections based on mode
    if mode == "Single Configuration":
        with st.sidebar.expander("FPN Parameters", expanded=True):
            lateral_weight = st.slider("Lateral Connection Weight", 0.0, 2.0, 1.0, 0.05,
                                       help="Strength of bottom-up features fused with top-down path. Higher values preserve fine spatial details.")
            upsample_method = st.selectbox("Upsampling Method", ['bilinear', 'nearest'],
                                          index=0, help="Bilinear gives smoother feature maps, nearest is faster but blockier.")
            confidence_threshold = st.slider("Confidence Threshold (simulated)", 0.0, 1.0, 0.5,
                                            help="Minimum activation to consider a detection (visualization only).")

        # Instantiate FPN
        fpn = FPNSimulator(image, lateral_weight=lateral_weight, upsampling_method=upsample_method)
        bottom_up = fpn.get_bottom_up()
        pyramid = fpn.get_pyramid()

        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["📊 Feature Maps", "🔍 Pyramid Outputs", "📈 Detection Analysis"])

        with tab1:
            st.subheader("Bottom-Up Features (Simulated Backbone)")
            st.caption("C2-C5: Different spatial resolutions & semantics. C2 has fine detail, C5 is coarse but semantically rich.")
            fig_bu = plot_feature_grid(bottom_up, "Bottom-Up Features (C2→C5)", colormap='inferno')
            st.pyplot(fig_bu)
            plt.close(fig_bu)

        with tab2:
            st.subheader("FPN Output Pyramids (P2-P5)")
            st.caption("After top-down fusion with lateral connections: each level now has strong semantics AND appropriate resolution.")
            fig_pyr = plot_feature_grid(pyramid, "FPN Output (P2→P5)", colormap='plasma')
            st.pyplot(fig_pyr)
            plt.close(fig_pyr)

        with tab3:
            st.subheader("Simulated Detection Performance")
            scores = fpn.compute_detection_scores()
            fig_scores = plot_detection_bar_chart(scores, None, labels=('FPN Config',))
            st.pyplot(fig_scores)
            plt.close(fig_scores)
            st.markdown("""
            *Interpretation:*  
            - **Small objects** benefit most from high-resolution P2 features enhanced by high-level semantics.  
            - **Large objects** are well-captured by coarser levels (P4/P5).  
            Adjust lateral weight to trade off between detail and context.
            """)

    else:  # Side-by-Side Comparison Mode
        st.sidebar.markdown("---")
        st.sidebar.subheader("Configuration A (Left)")
        lateral_weight_A = st.sidebar.slider("Lateral Weight - A", 0.0, 2.0, 0.3, 0.05, key='A')
        upsample_A = st.sidebar.selectbox("Upsampling - A", ['bilinear', 'nearest'], index=0, key='upA')

        st.sidebar.subheader("Configuration B (Right)")
        lateral_weight_B = st.sidebar.slider("Lateral Weight - B", 0.0, 2.0, 1.5, 0.05, key='B')
        upsample_B = st.sidebar.selectbox("Upsampling - B", ['bilinear', 'nearest'], index=1, key='upB')

        # Create two FPN instances
        fpn_left = FPNSimulator(image, lateral_weight=lateral_weight_A, upsampling_method=upsample_A)
        fpn_right = FPNSimulator(image, lateral_weight=lateral_weight_B, upsampling_method=upsample_B)

        st.subheader("🔄 Side-by-Side: Compare Two FPN Configurations")
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown("#### Configuration A")
            st.caption(f"Lateral weight = {lateral_weight_A}, Upsampling = {upsample_A}")
            fig_left = plot_feature_grid(fpn_left.get_pyramid(), "FPN Outputs", colormap='magma', figsize=(10, 2.5))
            st.pyplot(fig_left)
            plt.close(fig_left)

        with col_right:
            st.markdown("#### Configuration B")
            st.caption(f"Lateral weight = {lateral_weight_B}, Upsampling = {upsample_B}")
            fig_right = plot_feature_grid(fpn_right.get_pyramid(), "FPN Outputs", colormap='magma', figsize=(10, 2.5))
            st.pyplot(fig_right)
            plt.close(fig_right)

        # Direct comparison of fused feature maps
        st.markdown("### Direct Overlay Comparison")
        fig_comp = plot_comparison_side_by_side(fpn_left, fpn_right, ("Config A", "Config B"), colormap='coolwarm')
        st.pyplot(fig_comp)
        plt.close(fig_comp)

        # Detection scores comparison
        scores_A = fpn_left.compute_detection_scores()
        scores_B = fpn_right.compute_detection_scores()
        fig_bar = plot_detection_bar_chart(scores_A, scores_B, labels=('Config A (weak lateral)', 'Config B (strong lateral)'))
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        st.info("💡 **Observation**: Strong lateral connections (Config B) enhance small object detection but may introduce noise. Weak lateral connections lose fine detail. The optimal depends on your task.")

    # Reset button
    if st.sidebar.button("🔄 Reset to Defaults"):
        st.caching.clear_cache()
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **📘 Educational Notes**  
    - **Bottom-Up**: Standard CNN backbone (simulated).  
    - **Top-Down**: Upsamples coarser, semantic-rich features.  
    - **Lateral Connections**: Merge fine spatial detail from bottom-up.  
    - **FPN Output (P2-P5)**: Each level is both semantically strong and spatially precise.  
    *Reference: Lin et al., "Feature Pyramid Networks for Object Detection" (CVPR 2017).*
    """)


if __name__ == "__main__":
    main()
