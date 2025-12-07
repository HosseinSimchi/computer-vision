# app.py
import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork, AnchorGenerator
from torchvision.models.detection.image_list import ImageList
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
import io
import sys
from datetime import datetime
import glob

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Hossein Simchi - Computer Vision",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🔍"
)

# ---------------------------
# Configuration
# ---------------------------
GITHUB_REPO_URL = "https://github.com/HosseinSimchi/computer-vision"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "dataset/images"
ANNOT_DIR = "dataset/annotations"
TARGET_SIZE = (224, 224)
OUTPUT_DIR = "model_outputs"
RPN_PROPOSAL_DIR = os.path.join(OUTPUT_DIR, "rpn_proposals")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RPN_PROPOSAL_DIR, exist_ok=True)

# ---------------------------
# Custom CSS for beautiful styling (from your theme)
# ---------------------------
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2d3748;
        border-left: 5px solid #667eea;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        height: 100%;
        transition: transform 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Class badges */
    .class-badge {
        display: inline-block;
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Code block styling */
    .code-block {
        background: #2d3748;
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 8px;
        font-family: 'Monaco', 'Menlo', monospace;
        border-left: 4px solid #667eea;
    }
    
    /* Divider styling */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        margin: 2rem 0;
        border: none;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper: load dataset
# ---------------------------
@st.cache_resource
def load_dataset_cached(image_dir=IMAGE_DIR, annot_dir=ANNOT_DIR):
    dataset = []
    class_names = []
    if not os.path.isdir(image_dir) or not os.path.isdir(annot_dir):
        return dataset, class_names

    class_names = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    csv_files = sorted([f for f in os.listdir(annot_dir) if f.endswith(".csv")])

    # If counts mismatch, handle gracefully by matching file names by folder name if possible
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(image_dir, class_name)
        csv_path = None
        # try to find csv with the class_name in it
        for f in csv_files:
            if class_name in f:
                csv_path = os.path.join(annot_dir, f)
                break
        if csv_path is None and i < len(csv_files):
            csv_path = os.path.join(annot_dir, csv_files[i])

        if csv_path is None:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        for image_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, image_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w, _ = img.shape
            row = df[df.get("image_name", df.columns[0]) == image_name]
            if row.empty:
                continue

            ann = row.iloc[0, 1:].tolist()
            if len(ann) < 4:
                continue

            # scale boxes
            ann_scaled = [
                ann[0] / w * TARGET_SIZE[0], ann[1] / h * TARGET_SIZE[1],
                ann[2] / w * TARGET_SIZE[0], ann[3] / h * TARGET_SIZE[1]
            ]

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, TARGET_SIZE)

            img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
            label_tensor = torch.tensor([i], dtype=torch.int64)
            target = {"boxes": torch.tensor([ann_scaled], dtype=torch.float32), "labels": label_tensor}
            dataset.append((img_tensor, target))

    return dataset, class_names

# ---------------------------
# Helper: build models
# ---------------------------
@st.cache_resource
def build_models_cached():
    resnet = torchvision.models.resnet18(pretrained=False)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
    backbone.out_channels = 512
    for p in backbone.parameters():
        p.requires_grad = False

    anchor_gen = AnchorGenerator(sizes=((32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
    num_anchors = anchor_gen.num_anchors_per_location()[0]
    rpn_head = RPNHead(backbone.out_channels, num_anchors)
    rpn = RegionProposalNetwork(
        anchor_gen, rpn_head,
        fg_iou_thresh=0.7, bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={"training": 2000, "testing": 1000},
        post_nms_top_n={"training": 1000, "testing": 500},
        nms_thresh=0.7
    )
    # move to device
    return backbone.to(DEVICE), rpn.to(DEVICE)

# ---------------------------
# Helper: save model summary
# ---------------------------
def save_model_summary(backbone):
    buf = io.StringIO()
    real_stdout = sys.stdout
    try:
        sys.stdout = buf
        summary(backbone, (3, 224, 224))
    finally:
        sys.stdout = real_stdout

    path = os.path.join(OUTPUT_DIR, "model_summary.txt")
    with open(path, "w") as f:
        f.write(buf.getvalue())
    return path

# ---------------------------
# Helper: training function
# ---------------------------
def train_model(backbone, rpn, dataset, num_epochs=3, batch_size=8):
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Initialize dataset first.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    optimizer = torch.optim.Adam(rpn.parameters(), lr=0.001)
    log_path = os.path.join(OUTPUT_DIR, "training_log.txt")

    with open(log_path, "w") as log:
        for epoch in range(num_epochs):
            epoch_losses = []
            for images, targets in dataloader:
                optimizer.zero_grad()
                imgs_gpu = torch.stack([i.to(DEVICE) for i in images])
                targets_gpu = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                with torch.no_grad():
                    features = backbone(imgs_gpu)

                img_list = ImageList(imgs_gpu, [i.shape[-2:] for i in images])
                _, loss_dict = rpn(img_list, {"0": features}, targets_gpu)
                loss = loss_dict.get("loss_objectness", 0) + loss_dict.get("loss_rpn_box_reg", 0)

                if torch.isfinite(loss):
                    epoch_losses.append(loss.item())
                    loss.backward()
                    optimizer.step()

            if len(epoch_losses) > 0:
                mean_loss = float(np.mean(epoch_losses))
                line = f"Epoch {epoch+1}/{num_epochs} | Loss {mean_loss:.6f}"
            else:
                line = f"Epoch {epoch+1}/{num_epochs} | No valid loss"
            log.write(line + "\n")
    return log_path

# ---------------------------
# Helper: visualize proposals for uploaded image
# ---------------------------
def visualize_rpn_and_save(uploaded_file, backbone, rpn):
    # read file bytes and decode
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("Uploaded file could not be read as an image.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, TARGET_SIZE)

    tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        feat = backbone(tensor.unsqueeze(0))
        image_list = ImageList(tensor.unsqueeze(0), [tensor.shape[-2:]])
        proposals, _ = rpn(image_list, {"0": feat})

    top_proposals = proposals[0][:5].cpu().numpy()
    img_draw = img_resized.copy()

    for i, box in enumerate(top_proposals):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if i == 0 else (255, 255, 0) if i < 3 else (255, 0, 0)
        thickness = 3 if i == 0 else 2 if i < 3 else 1
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)

    # save with timestamp
    base_name = f"rpn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = os.path.join(RPN_PROPOSAL_DIR, base_name)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_draw)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return out_path

# ---------------------------
# Layout + UI Functions (theme sections)
# ---------------------------
def create_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">🚀 Computer Vision</div>', unsafe_allow_html=True)
        st.markdown("### Advanced Object Detection System")
        st.markdown("*Developed as part of the **DataYad Computer Vision Course***")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.success("🎯 Multi-task Learning")
        with col_b:
            st.info("📦 3 Object Classes")
        with col_c:
            st.success("⚡ Real-time Ready")
    st.markdown("---")

def create_project_overview(class_names):
    st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Object Detection")
        st.markdown("Multi-task learning model for simultaneous classification and localization")
        st.markdown("**Detected Classes:**")
        if class_names:
            for name in class_names[:5]:
                st.markdown(f'<span class="class-badge">{name}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="class-badge">No classes found</span>', unsafe_allow_html=True)
        st.markdown("**Data Pipeline:**")
        st.markdown("• Image normalization & resizing")
        st.markdown("• Bounding box coordinate scaling")
        st.markdown("• Real-time preprocessing")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Model Architecture")
        st.markdown("ResNet18 backbone + Region Proposal Network (RPN)")
        st.markdown("**Backbone:** ResNet18 (features → RPN)")
        st.markdown("**RPN:** Anchor generator + objectness & box-regression heads")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("• PyTorch & TorchVision")
        st.markdown("• OpenCV & Matplotlib")
        st.markdown("• Streamlit UI")
        st.markdown('</div>', unsafe_allow_html=True)

def create_arch_section():
    st.markdown('<div class="section-header">🏗️ Model Architecture</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Architecture Details")
        st.markdown("**Backbone:** ResNet18 truncated (no classifier) → feature map")
        st.markdown("**RPN:** Anchor sizes (32,64,128), aspect ratios (0.5,1.0,2.0)")
    with col2:
        st.markdown("### 📈 Quick Metrics")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**512**")
        st.markdown("Backbone channels")
        st.markdown('</div>', unsafe_allow_html=True)

def create_get_started_section():
    st.markdown('<div class="section-header">🚀 Get Started</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📚 Course Learnings")
        st.markdown("• Multi-task Learning")
        st.markdown("• Data Preprocessing & Scaling")
    with col2:
        st.markdown("### 💡 Pro Tips")
        st.markdown("• Use a GPU for training")
        st.markdown("• Monitor losses and checkpoints")

def create_source_section():
    st.markdown('<div class="section-header">💻 Source & Outputs</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background: #f8fafc; padding: 1rem; border-radius: 12px; border-left: 4px solid #667eea;'>
        <strong>Repository:</strong>
        <a href='{GITHUB_REPO_URL}' target='_blank'>{GITHUB_REPO_URL}</a>
    </div>
    """, unsafe_allow_html=True)

def create_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h3 style='color: #2d3748;'>Developed by Hossein Simchi</h3>
            <p style='color: #718096;'>DataYad Computer Vision Course Project</p>
            <a href='{}' target='_blank' style='
                display: inline-block;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 0.6rem 1.4rem;
                border-radius: 25px;
                text-decoration: none;
                font-weight: 600;
                margin-top: 0.5rem;
            '>⭐ Star on GitHub</a>
        </div>
        """.format(GITHUB_REPO_URL), unsafe_allow_html=True)

# ---------------------------
# Main Dashboard Area (Option B)
# ---------------------------
def main():
    create_header()
    # load dataset (cached) quickly for display
    dataset_preview, class_names = load_dataset_cached()
    create_project_overview(class_names)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    create_arch_section()
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    create_get_started_section()
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    create_source_section()

    st.markdown("")  # spacing

    # Dashboard cards row
    col_init, col_train, col_vis = st.columns(3, gap="large")

    # --------------------
    # Initialize Models Card
    # --------------------
    with col_init:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Initialize Dataset & Models")
        st.write("Load dataset from `dataset/images` and `dataset/annotations`, build ResNet backbone and RPN.")
        if st.button("🔁 Initialize / Refresh"):
            with st.spinner("Loading dataset and building models..."):
                ds, classes = load_dataset_cached()
                backbone, rpn = build_models_cached()
                sum_path = save_model_summary(backbone)
                # store in session
                st.session_state["dataset"] = ds
                st.session_state["class_names"] = classes
                st.session_state["backbone"] = backbone
                st.session_state["rpn"] = rpn
                st.session_state["model_summary_path"] = sum_path
                st.success("Models and dataset initialized.")
        # Show quick stats
        ds_len = len(st.session_state.get("dataset", dataset_preview))
        st.markdown(f"**Dataset items:** {ds_len}")
        st.markdown(f"**Device:** `{DEVICE}`")
        # show model summary preview if available
        summary_path = st.session_state.get("model_summary_path")
        if summary_path and os.path.exists(summary_path):
            if st.button("📄 Show model_summary.txt"):
                with open(summary_path, "r") as f:
                    txt = f.read()
                st.code(txt, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------
    # Training Card
    # --------------------
    with col_train:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Train RPN Model")
        st.write("Configure training parameters and start training. Training log will be saved to `model_outputs/training_log.txt`.")

        num_epochs = st.number_input("Epochs", min_value=1, max_value=50, value=3, step=1)
        batch_size = st.number_input("Batch size", min_value=1, max_value=32, value=8, step=1)
        train_btn = st.button("▶️ Start Training")

        if train_btn:
            if "backbone" not in st.session_state or "rpn" not in st.session_state or "dataset" not in st.session_state:
                st.error("Please initialize dataset & models first.")
            else:
                backbone = st.session_state["backbone"]
                rpn = st.session_state["rpn"]
                dataset = st.session_state["dataset"]
                log_placeholder = st.empty()
                progress_bar = st.progress(0)
                try:
                    # run training and stream progress by epochs
                    log_path = os.path.join(OUTPUT_DIR, "training_log.txt")
                    # ensure previous log removed
                    if os.path.exists(log_path):
                        os.remove(log_path)

                    for epoch in range(int(num_epochs)):
                        # run one-epoch training (simplified loop to provide feedback)
                        # We'll call train_model for all epochs but we want per-epoch feedback,
                        # so run a single-epoch mini training loop here
                        dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
                        optimizer = torch.optim.Adam(rpn.parameters(), lr=0.001)
                        epoch_losses = []
                        for images, targets in dataloader:
                            optimizer.zero_grad()
                            imgs_gpu = torch.stack([i.to(DEVICE) for i in images])
                            targets_gpu = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                            with torch.no_grad():
                                features = backbone(imgs_gpu)
                            img_list = ImageList(imgs_gpu, [i.shape[-2:] for i in images])
                            _, loss_dict = rpn(img_list, {"0": features}, targets_gpu)
                            loss = loss_dict.get("loss_objectness", 0) + loss_dict.get("loss_rpn_box_reg", 0)
                            if torch.isfinite(loss):
                                epoch_losses.append(loss.item())
                                loss.backward()
                                optimizer.step()
                        if len(epoch_losses) > 0:
                            mean_loss = float(np.mean(epoch_losses))
                            log_line = f"Epoch {epoch+1}/{num_epochs} | Loss: {mean_loss:.6f}"
                        else:
                            log_line = f"Epoch {epoch+1}/{num_epochs} | No valid loss"
                        # append to log file
                        with open(log_path, "a") as lf:
                            lf.write(log_line + "\n")
                        # update UI
                        log_placeholder.text(log_line)
                        progress_bar.progress(int((epoch + 1) / num_epochs * 100))
                    st.success("Training finished.")
                    st.session_state["training_log_path"] = log_path
                except Exception as e:
                    st.error(f"Training error: {e}")
        # Show training log and download
        if "training_log_path" in st.session_state:
            log_path = st.session_state["training_log_path"]
            if os.path.exists(log_path):
                if st.button("📄 Show training log"):
                    with open(log_path, "r") as f:
                        st.text(f.read())
                with open(log_path, "r") as f:
                    log_bytes = f.read().encode("utf-8")
                st.download_button("⬇️ Download training_log.txt", data=log_bytes, file_name="training_log.txt")
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------
    # Visualization Card
    # --------------------
    with col_vis:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Visualize RPN Proposals")
        st.write("Upload an image, run the RPN to get top proposals and save a visualization.")
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        vis_btn = st.button("🖼️ Generate & Save Proposals")

        if vis_btn:
            if "backbone" not in st.session_state or "rpn" not in st.session_state:
                st.error("Please initialize models first.")
            elif uploaded_file is None:
                st.error("Please upload an image.")
            else:
                try:
                    out_path = visualize_rpn_and_save(uploaded_file, st.session_state["backbone"], st.session_state["rpn"])
                    st.image(out_path, caption="RPN Proposals", use_column_width=True)
                    st.success(f"Saved proposals to: {out_path}")
                except Exception as e:
                    st.error(f"Visualization error: {e}")

        # List saved proposal images with download buttons
        st.markdown("**Saved proposals**")
        proposals = sorted(glob.glob(os.path.join(RPN_PROPOSAL_DIR, "*.png")), reverse=True)
        if proposals:
            for p in proposals[:10]:
                st.write(os.path.basename(p))
                st.image(p, width=220)
                with open(p, "rb") as f:
                    st.download_button("⬇️ Download", data=f, file_name=os.path.basename(p))
        else:
            st.markdown("_No proposals saved yet_")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # Bottom sections: files & source
    # ---------------------------
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    create_source_section()
    st.markdown("### 📁 Saved Files in `model_outputs`")
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*")), reverse=True)
    if files:
        for f in files:
            if os.path.isdir(f):
                st.write(f"📂 {os.path.basename(f)} (directory)")
                subfiles = glob.glob(os.path.join(f, "*"))
                for sf in subfiles[:10]:
                    st.write("•", os.path.basename(sf))
            else:
                st.write("•", os.path.basename(f))
                if os.path.exists(f):
                    with open(f, "rb") as fh:
                        st.download_button("⬇️ Download", data=fh, file_name=os.path.basename(f))
    else:
        st.write("No output files yet.")

    create_footer()

if __name__ == "__main__":
    main()
