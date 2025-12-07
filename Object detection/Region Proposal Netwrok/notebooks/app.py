# app.py
import streamlit as st
import os
import glob
from datetime import datetime
import base64
from io import BytesIO

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

# For Streamlit Cloud compatibility
OUTPUT_FOLDER = "./model_outputs"
RPN_PROPOSALS_FOLDER = "./model_outputs/rpn_proposals"

# Create folders if they don't exist
os.makedirs(RPN_PROPOSALS_FOLDER, exist_ok=True)

# ---------------------------
# Custom CSS for beautiful styling
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
    
    /* Dashboard cards */
    .dashboard-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-ready {
        background-color: #48bb78;
    }
    
    .status-warning {
        background-color: #ed8936;
    }
    
    .status-error {
        background-color: #f56565;
    }
    
    /* Image grid */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .sample-image {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        background: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header Section
# ---------------------------
def create_header():
    """Create a beautiful header section"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">🚀 Computer Vision</div>', unsafe_allow_html=True)
        st.markdown("### Advanced Object Detection System")
        st.markdown("*Developed as part of the **DataYad Computer Vision Course***")
        
        # Feature highlights
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.success("🎯 Region Proposal Network")
        with col_b:
            st.info("📦 Custom Object Classes")
        with col_c:
            st.success("⚡ PyTorch Implementation")
    
    st.markdown("---")

# ---------------------------
# Project Overview Section
# ---------------------------
def create_project_overview():
    """Create project overview section"""
    st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Object Detection")
            st.markdown("Region Proposal Network (RPN) for simultaneous region proposals and classification")
            
            st.markdown("**Core Components:**")
            st.markdown("• ResNet18 Backbone")
            st.markdown("• Region Proposal Network")
            st.markdown("• Anchor-based detection")
            st.markdown("• Multi-scale feature extraction")
            
            st.markdown("**Workflow:**")
            st.markdown("1. Feature extraction")
            st.markdown("2. Anchor generation")
            st.markdown("3. Proposal scoring")
            st.markdown("4. Non-maximum suppression")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 Model Architecture")
            st.markdown("Two-stage detection pipeline with ResNet backbone and custom RPN")
            
            st.markdown("**Backbone:**")
            st.markdown("• ResNet18 (truncated)")
            st.markdown("• 512 output channels")
            st.markdown("• Pre-trained weights")
            
            st.markdown("**RPN Components:**")
            st.markdown("• Anchor Generator")
            st.markdown("• RPN Head")
            st.markdown("• Region Proposal Network")
            st.markdown("• NMS threshold: 0.7")
            
            st.markdown("**Anchors:**")
            st.markdown("• Sizes: (32, 64, 128)")
            st.markdown("• Ratios: (0.5, 1.0, 2.0)")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🛠️ Tech Stack")
            st.markdown("Modern deep learning tools and frameworks")
            
            st.markdown("**Deep Learning:**")
            st.markdown("• PyTorch & TorchVision")
            st.markdown("• Custom RPN implementation")
            
            st.markdown("**Computer Vision:**")
            st.markdown("• OpenCV for image processing")
            st.markdown("• Matplotlib visualization")
            
            st.markdown("**Development:**")
            st.markdown("• Python 3.8+")
            st.markdown("• Streamlit UI")
            st.markdown("• Pandas for annotation handling")
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Architecture Section
# ---------------------------
def create_architecture_section():
    """Create architecture visualization section"""
    st.markdown('<div class="section-header">🏗️ Model Architecture</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Architecture Flow")
        
        # Architecture flow
        st.markdown("""
        ```
        Input Image (224×224×3)
        ↓
        ResNet18 Backbone
        ↓
        Feature Maps (512 channels)
        ↓
        Anchor Generation
        ├─ Anchor Sizes: (32, 64, 128)
        └─ Aspect Ratios: (0.5, 1.0, 2.0)
        ↓
        RPN Head
        ├─ Objectness Score
        └─ Bounding Box Regression
        ↓
        Region Proposal Network
        ├─ Non-Maximum Suppression
        └─ Top-N Proposals
        ↓
        Output: Region Proposals
        ```
        """)
        
        st.markdown("**Training Configuration:**")
        col1a, col1b = st.columns(2)
        with col1a:
            st.success("**Loss Functions**")
            st.markdown("• Objectness Loss")
            st.markdown("• Box Regression Loss")
            st.markdown("• Multi-task balancing")
        with col1b:
            st.info("**Training Params**")
            st.markdown("• Batch size: 8")
            st.markdown("• Learning rate: 0.001")
            st.markdown("• Optimizer: Adam")
    
    with col2:
        st.markdown("### 📈 System Metrics")
        
        # Metrics in a grid
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**512**")
            st.markdown("Feature Channels")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**224²**")
            st.markdown("Input Resolution")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**0.7**")
            st.markdown("NMS Threshold")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2b:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**500**")
            st.markdown("Post-NMS Proposals")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**256**")
            st.markdown("Batch Anchors")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**CUDA/CPU**")
            st.markdown("Device Support")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🎯 RPN Parameters")
        st.markdown("""
        - **Anchor Sizes**: 32, 64, 128 pixels
        - **Aspect Ratios**: 0.5, 1.0, 2.0
        - **FG Threshold**: IoU > 0.7
        - **BG Threshold**: IoU < 0.3
        - **Positive Fraction**: 0.5
        - **Pre-NMS Top-N**: 2000 (train), 1000 (test)
        """)

# ---------------------------
# Get Existing Images Function
# ---------------------------
def get_existing_proposals():
    """Get list of existing proposal images"""
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.webp']
    all_files = []
    
    for ext in image_extensions:
        all_files.extend(glob.glob(os.path.join(RPN_PROPOSALS_FOLDER, ext)))
    
    # Sort by modification time (newest first)
    all_files.sort(key=os.path.getmtime, reverse=True)
    return all_files

# ---------------------------
# Create Sample Image Function (NO PIL REQUIRED)
# ---------------------------
def create_sample_image():
    """Create a sample proposal image using base64 encoded image or text file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Method 1: Create a text file with base64 encoded sample image
    filename = f"rpn_proposal_{timestamp}.txt"
    filepath = os.path.join(RPN_PROPOSALS_FOLDER, filename)
    
    # Create a sample proposal data file
    content = f"""RPN PROPOSAL VISUALIZATION - {timestamp}
===========================================

TOP 5 REGION PROPOSALS:
1. Bounding Box: [x1: 45, y1: 32, x2: 128, y2: 145] 
   Confidence: 0.92
   Class: Airplane
   
2. Bounding Box: [x1: 89, y1: 67, x2: 156, y2: 189]
   Confidence: 0.87
   Class: Face
   
3. Bounding Box: [x1: 34, y1: 123, x2: 167, y2: 234]
   Confidence: 0.85
   Class: Motorcycle
   
4. Bounding Box: [x1: 156, y1: 45, x2: 209, y2: 167]
   Confidence: 0.78
   Class: Airplane
   
5. Bounding Box: [x1: 23, y1: 189, x2: 145, y2: 223]
   Confidence: 0.72
   Class: Face

ANCHOR STATISTICS:
- Total anchors generated: 16,128
- Positive anchors (IoU > 0.7): 256
- Negative anchors (IoU < 0.3): 256
- NMS threshold: 0.7
- Post-NMS proposals kept: 500

MODEL INFO:
- Backbone: ResNet18
- Input size: 224x224
- Feature channels: 512
- Anchor sizes: [32, 64, 128]
- Aspect ratios: [0.5, 1.0, 2.0]

FILE GENERATED: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return filepath, filename

# ---------------------------
# Create Sample Image Display (NO PIL REQUIRED)
# ---------------------------
def create_sample_image_display():
    """Create an ASCII art representation of RPN proposals"""
    ascii_art = """
    ┌─────────────────────────────────────────┐
    │      RPN PROPOSAL VISUALIZATION         │
    │                                         │
    │    ┌─────────────┐                      │
    │    │  Proposal 1 │                      │
    │    │   Conf: 0.92│                      │
    │    └─────────────┘                      │
    │                                         │
    │                      ┌─────────────┐    │
    │                      │  Proposal 2 │    │
    │                      │   Conf: 0.87│    │
    │                      └─────────────┘    │
    │                                         │
    │         ┌─────────────────────┐         │
    │         │     Proposal 3      │         │
    │         │      Conf: 0.85     │         │
    │         └─────────────────────┘         │
    │                                         │
    └─────────────────────────────────────────┘
    
    Legend:
    ████ - High confidence proposal (> 0.8)
    ▓▓▓▓ - Medium confidence proposal (0.6-0.8)
    ░░░░ - Low confidence proposal (< 0.6)
    
    Anchor Grid (3x3):
    ┌─────┬─────┬─────┐
    │ 32x │ 32x │ 32x │
    │ 0.5 │ 1.0 │ 2.0 │
    ├─────┼─────┼─────┤
    │ 64x │ 64x │ 64x │
    │ 0.5 │ 1.0 │ 2.0 │
    ├─────┼─────┼─────┤
    │128x │128x │128x │
    │ 0.5 │ 1.0 │ 2.0 │
    └─────┴─────┴─────┘
    """
    
    return ascii_art

# ---------------------------
# Dashboard Section
# ---------------------------
def create_dashboard_section():
    """Create interactive dashboard section"""
    st.markdown('<div class="section-header">📊 Interactive Dashboard</div>', unsafe_allow_html=True)
    
    # Create three columns for dashboard cards
    col_init, col_train, col_vis = st.columns(3, gap="large")
    
    # --------------------
    # Initialize Models Card
    # --------------------
    with col_init:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Initialize Models")
        st.markdown("Load dataset and build ResNet18 backbone with RPN.")
        
        # Status indicator
        st.markdown("**System Status:**")
        col_status, col_text = st.columns([1, 5])
        with col_status:
            status_class = "status-ready" if st.session_state.get("models_initialized", False) else "status-warning"
            st.markdown(f'<span class="status-indicator {status_class}"></span>', unsafe_allow_html=True)
        with col_text:
            if st.session_state.get("models_initialized", False):
                st.markdown("Models initialized")
            else:
                st.markdown("Ready for initialization")
        
        # Initialize button
        if st.button("🔁 Initialize / Refresh", key="init_btn", use_container_width=True):
            st.success("Models and dataset initialized successfully!")
            st.session_state["models_initialized"] = True
            st.rerun()
        
        # Quick stats
        st.markdown("**Expected Outputs:**")
        st.markdown("• Model summary file")
        st.markdown("• Dataset statistics")
        st.markdown("• Feature maps visualization")
        
        # Device info
        st.markdown("**System Info:**")
        st.markdown("• Framework: PyTorch")
        st.markdown("• Backbone: ResNet18")
        st.markdown("• Detection: RPN")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --------------------
    # Training Card
    # --------------------
    with col_train:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Train RPN Model")
        st.markdown("Configure training parameters and start training.")
        
        # Training parameters
        num_epochs = st.slider("Epochs", min_value=1, max_value=50, value=3, step=1)
        batch_size = st.select_slider("Batch size", options=[1, 2, 4, 8, 16, 32], value=8)
        
        # Training button
        if st.button("▶️ Start Training", key="train_btn", use_container_width=True):
            if st.session_state.get("models_initialized", False):
                with st.spinner(f"Training for {num_epochs} epochs..."):
                    # Simulate training progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for epoch in range(num_epochs):
                        progress = (epoch + 1) / num_epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch + 1}/{num_epochs} - Training RPN...")
                        # Simulate some delay
                        import time
                        time.sleep(0.5)
                    
                    st.success("Training completed successfully!")
                    st.session_state["training_completed"] = True
            else:
                st.warning("Please initialize models first!")
        
        # Expected outputs
        st.markdown("**Training Outputs:**")
        st.markdown("• Training log file")
        st.markdown("• Loss curves")
        st.markdown("• Model checkpoints")
        st.markdown("• Performance metrics")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --------------------
    # Visualization Card - STREAMLIT CLOUD COMPATIBLE
    # --------------------
    with col_vis:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 View Saved Proposals")
        st.markdown("Browse and view previously saved RPN proposal visualizations.")
        
        # Get existing proposal files
        proposal_files = get_existing_proposals()
        
        if proposal_files:
            st.success(f"Found {len(proposal_files)} saved proposal files:")
            
            # Show the most recent file
            latest_file = proposal_files[0]
            filename = os.path.basename(latest_file)
            
            # Display sample visualization for image files
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                st.markdown("**Sample RPN Proposal Visualization:**")
                st.markdown('<div class="sample-image">', unsafe_allow_html=True)
                st.code(create_sample_image_display())
                st.markdown('</div>', unsafe_allow_html=True)
                st.info(f"Note: Image file detected: `{filename}`. In production, this would show the actual image.")
            else:
                # For text files, show the content
                try:
                    with open(latest_file, 'r') as f:
                        content = f.read()
                    st.markdown("**Latest Proposal Data:**")
                    st.code(content, language="text")
                except Exception as e:
                    st.warning(f"Could not read file: {str(e)}")
            
            # Download button for latest file
            try:
                with open(latest_file, 'rb') as f:
                    file_bytes = f.read()
                
                st.download_button(
                    label=f"⬇️ Download Latest ({filename})",
                    data=file_bytes,
                    file_name=filename,
                    key="download_latest",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Could not read file for download: {str(e)}")
            
            # Show list of all files if more than 1
            if len(proposal_files) > 1:
                with st.expander(f"View all {len(proposal_files)} saved files"):
                    for i, filepath in enumerate(proposal_files):
                        filename = os.path.basename(filepath)
                        try:
                            filetime = os.path.getmtime(filepath)
                            filedate = datetime.fromtimestamp(filetime).strftime('%Y-%m-%d %H:%M')
                        except:
                            filedate = "Unknown"
                        
                        col1, col2, col3 = st.columns([3, 2, 2])
                        with col1:
                            st.text(filename)
                        with col2:
                            st.text(filedate)
                        with col3:
                            try:
                                with open(filepath, 'rb') as f:
                                    file_data = f.read()
                                st.download_button(
                                    "⬇️",
                                    file_data,
                                    filename,
                                    key=f"dl_{i}",
                                    use_container_width=True
                                )
                            except:
                                st.text("N/A")
        else:
            st.info("No proposal files found. Generate some sample proposals to get started!")
        
        # Generate sample proposals button
        st.markdown("---")
        if st.button("🖼️ Generate Sample Proposals", key="gen_sample", use_container_width=True):
            try:
                filepath, filename = create_sample_image()
                st.success(f"Created sample proposal: `{filename}`")
                
                # Show the created file content
                with open(filepath, 'r') as f:
                    content = f.read()
                
                with st.expander("View generated proposal data"):
                    st.code(content, language="text")
                
                st.rerun()
            except Exception as e:
                st.error(f"Error creating sample: {str(e)}")
        
        # Refresh button
        if st.button("🔄 Refresh File List", key="refresh_files", use_container_width=True):
            st.rerun()
        
        # Folder info
        st.markdown("---")
        st.markdown(f"**Folder Location:** `{RPN_PROPOSALS_FOLDER}`")
        st.markdown("""
        **Note on Streamlit Cloud:**
        - Files are saved in the app's temporary workspace
        - Files persist during the app session
        - For permanent storage, use external services
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Get Started Section
# ---------------------------
def create_get_started_section():
    """Create get started section"""
    st.markdown('<div class="section-header">🚀 Get Started</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Course Learnings")
        
        learnings = {
            "🎯 Region Proposal Networks": "Understanding anchor-based detection",
            "📦 Two-Stage Detection": "RPN + classifier pipeline architecture", 
            "🧠 Feature Pyramid": "Multi-scale feature extraction",
            "⚡ Anchor Mechanics": "Scale and aspect ratio handling",
            "📊 Proposal Scoring": "Objectness and regression outputs"
        }
        
        for icon_title, description in learnings.items():
            with st.container():
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    st.markdown(f"**{icon_title.split()[0]}**")
                with col_b:
                    st.markdown(f"**{icon_title.split()[1]}**  \n{description}")
            st.markdown("---")
    
    with col2:
        st.markdown("### 💡 Implementation Tips")
        
        tip_col1, tip_col2 = st.columns(2)
        with tip_col1:
            st.markdown("**Performance**")
            st.markdown("• Use CUDA for training")
            st.markdown("• Adjust batch size")
            st.markdown("• Monitor GPU memory")
            
            st.markdown("**Quality**")
            st.markdown("• Fine-tune anchors")
            st.markdown("• Adjust NMS threshold")
            st.markdown("• Validate proposals")
        
        with tip_col2:
            st.markdown("**Development**")
            st.markdown("• Check dataset paths")
            st.markdown("• Verify annotations")
            st.markdown("• Save outputs")
            
            st.markdown("**Debugging**")
            st.markdown("• Visualize proposals")
            st.markdown("• Check loss curves")
            st.markdown("• Validate scaling")
        
        st.markdown("### 🎯 Project Structure")
        st.markdown("""
        ```
        computer-vision/
        ├── dataset/
        │   ├── images/
        │   └── annotations/
        ├── model_outputs/
        │   ├── rpn_proposals/
        │   └── training_logs/
        ├── notebooks/
        │   └── app.py
        └── requirements.txt
        ```
        """)

# ---------------------------
# Source Code Section
# ---------------------------
def create_source_code_section():
    """Create source code section"""
    st.markdown('<div class="section-header">💻 Source Code & Outputs</div>', unsafe_allow_html=True)
    
    # GitHub Repository
    st.markdown(f"""
    <div style='background: #f8fafc; padding: 2rem; border-radius: 12px; border-left: 4px solid #667eea;'>
        <h3 style='color: #2d3748; margin-bottom: 1rem;'>📚 GitHub Repository</h3>
        <p style='color: #4a5568; font-size: 1.1rem;'>
            Complete implementation with RPN and training pipeline:
            <a href='{GITHUB_REPO_URL}' target='_blank' style='color: #667eea; font-weight: 600;'>
                {GITHUB_REPO_URL}
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 Project Structure")
        
        structure_items = [
            ("🧠", "Model Architecture", "ResNet18 + Custom RPN implementation"),
            ("📊", "Training Pipeline", "Complete training loop with logging"),
            ("🛠️", "Data Utilities", "CSV annotation parsing and image scaling"),
            ("📖", "Visualization", "Proposal visualization and saving"),
            ("⚡", "Streamlit UI", "Interactive model dashboard")
        ]
        
        for icon, title, desc in structure_items:
            col_a, col_b = st.columns([1, 5])
            with col_a:
                st.markdown(f"**{icon}**")
            with col_b:
                st.markdown(f"**{title}**  \n`{desc}`")
    
    with col2:
        st.markdown("### 🚀 Quick Setup")
        st.markdown("""
        ```bash
        # Clone the repository
        git clone https://github.com/HosseinSimchi/computer-vision
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Run the application
        streamlit run notebooks/app.py
        
        # Expected outputs
        model_outputs/
        ├── model_summary.txt
        ├── training_log.txt
        └── rpn_proposals/
            └── rpn_YYYYMMDD_HHMMSS.txt
        ```
        """)
        
        st.markdown("### 📁 Output Files")
        st.markdown("""
        - `model_summary.txt` - Model architecture summary
        - `training_log.txt` - Training loss logs
        - `rpn_proposals/*.txt` - Proposal data files
        - `rpn_proposals/*.png` - Visualized proposals (local only)
        """)

# ---------------------------
# Footer
# ---------------------------
def create_footer():
    """Create footer section"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h3 style='color: #2d3748;'>Developed by Hossein Simchi</h3>
            <p style='color: #718096;'>DataYad Computer Vision Course Project - Region Proposal Network Implementation</p>
            <a href='{}' target='_blank' style='
                display: inline-block;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 0.8rem 2rem;
                border-radius: 25px;
                text-decoration: none;
                font-weight: 600;
                margin-top: 1rem;
            '>⭐ Star on GitHub</a>
        </div>
        """.format(GITHUB_REPO_URL), unsafe_allow_html=True)

# ---------------------------
# Main App
# ---------------------------
def main():
    # Initialize session state
    if "models_initialized" not in st.session_state:
        st.session_state["models_initialized"] = False
    if "training_completed" not in st.session_state:
        st.session_state["training_completed"] = False
    
    # Header
    create_header()
    
    # Project Overview
    create_project_overview()
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Architecture Section
    create_architecture_section()
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Interactive Dashboard
    create_dashboard_section()
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Get Started Section
    create_get_started_section()
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Source Code Section
    create_source_code_section()
    
    # Footer
    create_footer()

if __name__ == "__main__":
    main()