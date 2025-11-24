# Direct Download

```bash
git clone https://huggingface.co/Neurazum/Vbai-3D-1.0
```

# Vbai-3D 1.0

| Size | Params | F1 Score | mAPᵛᵃᴵ | Accuracy | ROC-AUC |
|-------|--------|-------|--------|--------|---------|
| _80x80x80_ | 8.32M | 68.27% | 51.28% | 51.82% | 54.39% |

## Description

Vbai-3D version 1.0 diagnoses dementia by scanning the brain in 3D from MRI files using tissue scanning. This model can also be used in real time by scanning the brain slice by slice.

#### Audience / Target

Vbai-3D models are developed exclusively for hospitals, health centers and science centers.

#### Classes

 - **AD**: The person has Alzheimer's disease..
 - **MCI**: The person has a cognitive impairment.
 - **CN**: The person does not have dementia.

# Usage

```bash
streamlit run test.py
```

**test.py:**
```python
"""
Vbai-3D 1.0 Real-Time MRI Monitoring System
Streamlit-based 3D MRI slice-by-slice visualization and AI prediction system

Usage:
    streamlit run {this_file}.py

Features:
    - 3D MRI (.nii/.nii.gz) file upload
    - Slice-by-slice visualization (Axial, Coronal, Sagittal)
    - Real-time AI prediction (CN, MCI, AD)
    - Probability distributions
    - Interactive visualization
    - Multi-view mode
"""

import streamlit as st
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os
import time


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ImprovedMRINet(nn.Module):
    def __init__(self, num_classes=3, in_channels=1):
        super(ImprovedMRINet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.se1 = SEBlock3D(64)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.se2 = SEBlock3D(128)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.se3 = SEBlock3D(256)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResBlock3D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock3D(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)

        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)

        x = self.fc(x)
        return x


def load_and_preprocess_nifti(file_path, target_shape=(80, 80, 80)):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
    except Exception as e:
        st.error(f"File loading error: {e}")
        return None

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    brain_mask = data > data.mean()

    if brain_mask.sum() > 0:
        brain_pixels = data[brain_mask]
        p1, p99 = np.percentile(brain_pixels, [1, 99])
        data = np.clip(data, p1, p99)

        mean = brain_pixels.mean()
        std = brain_pixels.std()
        if std > 1e-6:
            data = (data - mean) / (std + 1e-8)
        else:
            data = data - mean
    else:
        mean = data.mean()
        std = data.std()
        if std > 1e-6:
            data = (data - mean) / (std + 1e-8)
        else:
            data = data - mean

    data_min, data_max = data.min(), data.max()
    if abs(data_max - data_min) > 1e-6:
        data = (data - data_min) / (data_max - data_min + 1e-8)
    else:
        data = np.zeros_like(data)

    data = np.clip(data, 0, 1)
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)

    return data


def resize_volume(volume, target_shape):
    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(volume_tensor, size=target_shape,
                           mode='trilinear', align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedMRINet(num_classes=3).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, device


def predict_mri(model, device, volume, target_shape=(80, 80, 80)):
    if volume.shape != target_shape:
        volume = resize_volume(volume, target_shape)

    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(volume_tensor)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1)

    return pred.item(), probs.cpu().numpy()[0]


def create_slice_image(slice_2d, colormap='gray'):
    slice_norm = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8) * 255).astype(np.uint8)

    if colormap == 'gray':
        return Image.fromarray(slice_norm, mode='L')
    else:
        cmap = plt.get_cmap(colormap)
        colored = cmap(slice_norm / 255.0)
        return Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))


def plot_probability_bars(probs, class_names):
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.barh(class_names, probs, color=colors)

    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{prob*100:.2f}%',
               ha='left', va='center', fontweight='bold', fontsize=12)

    ax.set_xlim([0, 1])
    ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Class Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    return fig


def main():
    st.set_page_config(
        page_title="Vbai-3D 1.0 Monitoring",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧠 Vbai-3D 1.0 - Real-Time MRI Monitoring System")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")

        model_path = st.text_input(
            "Model Path",
            value="Vbai-3D 1.0.pth/model/path"
        )

        if st.button("🔄 Load Model"):
            with st.spinner("Loading model..."):
                st.session_state.model, st.session_state.device = load_model(model_path)
                if st.session_state.model is not None:
                    st.success("✅ Model loaded successfully!")
                    st.info(f"Device: {st.session_state.device}")

        st.markdown("---")

        st.subheader("🎨 Visualization")
        colormap = st.selectbox(
            "Color Palette",
            ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool']
        )

        multi_view = st.checkbox("Multi-View Mode", value=False)

        st.markdown("---")

        st.subheader("📐 Model Parameters")
        target_shape = (80, 80, 80)
        st.info(f"Target Size: {target_shape}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📂 File Upload")

        uploaded_file = st.file_uploader(
            "Upload 3D MRI file (.nii or .nii.gz)",
            type=['nii', 'nii.gz'],
            help="Select a NIfTI format 3D MRI file"
        )

    with col2:
        st.header("ℹ️ Information")
        st.info("""
        **Supported Classes:**
        - 🟢 CN: Cognitively Normal
        - 🟡 MCI: Mild Cognitive Impairment
        - 🔴 AD: Alzheimer's Disease
        """)

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            st.success(f"✅ File uploaded: {uploaded_file.name}")

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Reading file...")
            progress_bar.progress(20)
            data = load_and_preprocess_nifti(tmp_path, target_shape)

            if data is None:
                st.error("File could not be loaded!")
                st.stop()

            progress_bar.progress(40)
            status_text.text("Preprocessing data...")

            st.info(f"📊 Data Size: {data.shape}")

            if 'model' in st.session_state and st.session_state.model is not None:
                status_text.text("Running AI prediction...")
                progress_bar.progress(60)

                start_time = time.time()
                pred_class, probs = predict_mri(
                    st.session_state.model,
                    st.session_state.device,
                    data,
                    target_shape
                )
                inference_time = time.time() - start_time

                progress_bar.progress(80)

                class_names = ['CN (Normal)', 'MCI (Mild)', 'AD (Alzheimer)']
                class_colors = ['🟢', '🟡', '🔴']

                st.markdown("---")
                st.header("🎯 AI Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Predicted Class",
                        f"{class_colors[pred_class]} {class_names[pred_class]}"
                    )

                with col2:
                    st.metric(
                        "Confidence Score",
                        f"{probs[pred_class]*100:.2f}%"
                    )

                with col3:
                    st.metric(
                        "Prediction Time",
                        f"{inference_time:.3f} sec"
                    )

                st.subheader("📊 Class Probabilities")
                fig = plot_probability_bars(probs, class_names)
                st.pyplot(fig)

                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = -np.log(1.0 / 3)
                uncertainty = entropy / max_entropy

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("1st Choice", f"{class_names[np.argsort(probs)[-1]]}")
                with col2:
                    st.metric("2nd Choice", f"{class_names[np.argsort(probs)[-2]]}")
                with col3:
                    st.metric("Uncertainty", f"{uncertainty:.3f}")

                if uncertainty > 0.5:
                    st.warning("⚠️ Model is uncertain! Indecisive between different classes.")
                else:
                    st.success("✅ Model made a confident prediction.")

            else:
                st.warning("⚠️ Model not loaded. Please load the model from sidebar.")

            progress_bar.progress(100)
            status_text.text("Ready!")

            st.markdown("---")
            st.header("🔍 Slice Visualization")

            if multi_view:
                st.subheader("Multi-View (Axial, Coronal, Sagittal)")

                col1, col2, col3 = st.columns(3)
                with col1:
                    axial_idx = st.slider("Axial (Z)", 0, data.shape[2]-1, data.shape[2]//2)
                with col2:
                    coronal_idx = st.slider("Coronal (Y)", 0, data.shape[1]-1, data.shape[1]//2)
                with col3:
                    sagittal_idx = st.slider("Sagittal (X)", 0, data.shape[0]-1, data.shape[0]//2)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Axial (Z-axis)**")
                    axial_slice = data[:, :, axial_idx]
                    axial_img = create_slice_image(axial_slice, colormap)
                    st.image(axial_img, caption=f"Axial Slice #{axial_idx}", use_container_width=True)

                with col2:
                    st.markdown("**Coronal (Y-axis)**")
                    coronal_slice = data[:, coronal_idx, :]
                    coronal_img = create_slice_image(coronal_slice, colormap)
                    st.image(coronal_img, caption=f"Coronal Slice #{coronal_idx}", use_container_width=True)

                with col3:
                    st.markdown("**Sagittal (X-axis)**")
                    sagittal_slice = data[sagittal_idx, :, :]
                    sagittal_img = create_slice_image(sagittal_slice, colormap)
                    st.image(sagittal_img, caption=f"Sagittal Slice #{sagittal_idx}", use_container_width=True)

            else:
                axis = st.radio(
                    "Select Slice Axis",
                    ['Axial (Z)', 'Coronal (Y)', 'Sagittal (X)'],
                    horizontal=True
                )

                axis_map = {
                    'Axial (Z)': 2,
                    'Coronal (Y)': 1,
                    'Sagittal (X)': 0
                }
                axis_idx = axis_map[axis]

                slice_idx = st.slider(
                    "Slice Index",
                    0,
                    data.shape[axis_idx] - 1,
                    data.shape[axis_idx] // 2,
                    help=f"Select a value between 0 and {data.shape[axis_idx]-1}"
                )

                if axis_idx == 2:
                    slice_2d = data[:, :, slice_idx]
                elif axis_idx == 1:
                    slice_2d = data[:, slice_idx, :]
                else:
                    slice_2d = data[slice_idx, :, :]

                slice_img = create_slice_image(slice_2d, colormap)

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.image(
                        slice_img,
                        caption=f"{axis} - Slice #{slice_idx}",
                        use_container_width=True
                    )

                with col2:
                    st.markdown("**Statistics**")
                    st.metric("Min", f"{slice_2d.min():.3f}")
                    st.metric("Max", f"{slice_2d.max():.3f}")
                    st.metric("Mean", f"{slice_2d.mean():.3f}")
                    st.metric("Std", f"{slice_2d.std():.3f}")

            st.markdown("---")
            st.subheader("💾 Download Options")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📊 Generate Report"):
                    try:
                        pred_class_name = class_names[pred_class] if 'pred_class' in locals() and pred_class is not None else 'N/A'
                        pred_confidence = f"{probs[pred_class]*100:.2f}" if 'probs' in locals() and 'pred_class' in locals() and probs is not None and pred_class is not None else 'N/A'
                        uncertainty_value = f"{uncertainty:.3f}" if 'uncertainty' in locals() and uncertainty is not None else 'N/A'
                        prob_cn = f"{probs[0]*100:.2f}" if 'probs' in locals() and probs is not None else 'N/A'
                        prob_mci = f"{probs[1]*100:.2f}" if 'probs' in locals() and probs is not None else 'N/A'
                        prob_ad = f"{probs[2]*100:.2f}" if 'probs' in locals() and probs is not None else 'N/A'

                        report = f"""
VBAI-3D 1.0 - MRI Analysis Report
================================

File: {uploaded_file.name}
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}

Prediction Results:
-----------------
Class: {pred_class_name}
Confidence: {pred_confidence}%
Uncertainty: {uncertainty_value}

Probabilities:
-----------
CN (Normal): {prob_cn}%
MCI (Mild): {prob_mci}%
AD (Alzheimer): {prob_ad}%

Data Information:
--------------
Size: {data.shape}
Min: {data.min():.3f}
Max: {data.max():.3f}
Mean: {data.mean():.3f}
                        """
                        st.download_button(
                            label="📥 Download Report",
                            data=report,
                            file_name="mri_report.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error generating report: {e}")

            progress_bar.empty()
            status_text.empty()

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    else:
        st.info("👆 Please upload an MRI file")

        with st.expander("📖 User Guide"):
            st.markdown("""
            ### Steps:
            1. **Load model from sidebar**
            2. **Upload your 3D MRI file (.nii/.nii.gz)**
            3. **AI prediction will be done automatically**
            4. **Use sliders to examine slices**
            5. **Generate report if needed**

            ### Supported Formats:
            - .nii (NIfTI)
            - .nii.gz (Compressed NIfTI)

            ### Features:
            - Real-time AI prediction
            - 3-axis visualization (Axial, Coronal, Sagittal)
            - Multi-view mode
            - Probability analysis
            - Uncertainty calculation
            - Report generation
            """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Vbai-3D 1.0 | Powered by PyTorch & Streamlit | 2025"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
```
