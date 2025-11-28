# app.py
import streamlit as st

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
            st.success("🎯 Multi-task Learning")
        with col_b:
            st.info("📦 3 Object Classes")
        with col_c:
            st.success("⚡ Real-time Ready")
    
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
            st.markdown("Multi-task learning model for simultaneous classification and localization")
            
            st.markdown("**Detected Classes:**")
            st.markdown('<span class="class-badge">✈️ Airplane</span>', unsafe_allow_html=True)
            st.markdown('<span class="class-badge">😊 Face</span>', unsafe_allow_html=True)
            st.markdown('<span class="class-badge">🏍️ Motorcycle</span>', unsafe_allow_html=True)
            
            st.markdown("**Data Pipeline:**")
            st.markdown("• Image normalization & resizing")
            st.markdown("• Bounding box coordinate scaling")
            st.markdown("• Real-time preprocessing")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 Model Architecture")
            st.markdown("Custom CNN with dual output heads for robust detection")
            
            st.markdown("**Architecture:**")
            st.markdown("• 5 Conv2D + MaxPool layers")
            st.markdown("• Dual output heads")
            st.markdown("• 7.9M parameters")
            st.markdown("• Dropout regularization")
            
            st.markdown("**Training:**")
            st.markdown("• Adam optimizer")
            st.markdown("• Multi-task loss")
            st.markdown("• Early stopping")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🛠️ Tech Stack")
            st.markdown("Modern deep learning tools and frameworks")
            
            st.markdown("**Deep Learning:**")
            st.markdown("• TensorFlow & Keras")
            st.markdown("• Custom CNN architecture")
            
            st.markdown("**Computer Vision:**")
            st.markdown("• OpenCV processing")
            st.markdown("• PIL image handling")
            
            st.markdown("**Development:**")
            st.markdown("• Python 3.8+")
            st.markdown("• Streamlit UI")
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Architecture Section
# ---------------------------
def create_architecture_section():
    """Create architecture visualization section"""
    st.markdown('<div class="section-header">🏗️ Model Architecture</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Architecture Details")
        
        # Architecture flow
        st.markdown("**Layer Progression:**")
        st.markdown("""
        ```
        Input (224×224×3)
        ↓
        Conv2D(32) → MaxPool
        ↓
        Conv2D(64) → MaxPool  
        ↓
        Conv2D(128) → MaxPool
        ↓
        Conv2D(256) → MaxPool
        ↓
        Conv2D(512) → MaxPool
        ↓
        Flatten
        ↓
        Dense(128) → Dropout(0.5)
        ↘                       ↙
        Classification       Bounding Box
        (3 units)           (4 units)
        ```
        """)
        
        st.markdown("**Output Heads:**")
        col1a, col1b = st.columns(2)
        with col1a:
            st.success("**Classification**")
            st.markdown("• 3 output units")
            st.markdown("• Softmax activation")
            st.markdown("• Object categories")
        with col1b:
            st.info("**Bounding Box**")
            st.markdown("• 4 output units")
            st.markdown("• Sigmoid activation")
            st.markdown("• [x1, y1, x2, y2]")
    
    with col2:
        st.markdown("### 📈 Performance Metrics")
        
        # Metrics in a grid
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**7.9M**")
            st.markdown("Parameters")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**95%+**")
            st.markdown("Training Accuracy")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**224²**")
            st.markdown("Input Size")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2b:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**30.5 MB**")
            st.markdown("Model Size")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**90%+**")
            st.markdown("Validation Acc")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**~50ms**")
            st.markdown("Inference Time")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🎯 Training Configuration")
        st.markdown("""
        - **Optimizer**: Adam (lr=0.001)
        - **Epochs**: 100 with early stopping
        - **Batch Size**: 32
        - **Validation**: 15% split
        - **Callbacks**: Checkpointing, LR reduction
        """)

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
            "🎯 Multi-task Learning": "Simultaneous classification and localization",
            "📦 Data Processing": "Image normalization and annotation handling", 
            "🧠 CNN Architecture": "Custom model design with dual outputs",
            "⚡ Training Strategies": "Loss weighting and optimization",
            "📊 Evaluation": "Performance metrics and visualization"
        }
        
        for icon_title, description in learnings.items():
            with st.container():
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    st.markdown(f"**{icon_title}**")
                with col_b:
                    st.markdown(description)
            st.markdown("---")
    
    with col2:
        st.markdown("### 💡 Pro Tips")
        tip_col1, tip_col2 = st.columns(2)
        with tip_col1:
            st.markdown("• Experiment with hyperparameters")
            st.markdown("• Try data augmentation")
        with tip_col2:
            st.markdown("• Adjust loss weights")
            st.markdown("• Add regularization")
        
        st.markdown("### 🎯 Use Cases")
        st.markdown("• Object detection in images")
        st.markdown("• Real-time video analysis")
        st.markdown("• Educational purposes")
        st.markdown("• Research experiments")

# ---------------------------
# Source Code Section
# ---------------------------
def create_source_code_section():
    """Create source code section"""
    st.markdown('<div class="section-header">💻 Source Code</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: #f8fafc; padding: 2rem; border-radius: 12px; border-left: 4px solid #667eea;'>
        <h3 style='color: #2d3748; margin-bottom: 1rem;'>📚 GitHub Repository</h3>
        <p style='color: #4a5568; font-size: 1.1rem;'>
            Complete implementation available at: 
            <a href='{GITHUB_REPO_URL}' target='_blank' style='color: #667eea; font-weight: 600;'>
                {GITHUB_REPO_URL}
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 What's Included")
        features = [
            ("🧠", "Complete Model", "CNN architecture with dual outputs"),
            ("📊", "Training Scripts", "Full training pipeline with callbacks"),
            ("🛠️", "Utilities", "Data loading and visualization tools"),
            ("📖", "Documentation", "Course-derived implementations")
        ]
        
        for icon, title, desc in features:
            col_a, col_b = st.columns([1, 5])
            with col_a:
                st.markdown(f"**{icon}**")
            with col_b:
                st.markdown(f"**{title}**  \n{desc}")
    
    with col2:
        st.markdown("### 🚀 Quick Commands")
        st.markdown("""
        ```bash
        # Get the code
        git clone https://github.com/HosseinSimchi/computer-vision
        
        # Install dependencies  
        pip install -r requirements.txt
        
        ```
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
            <p style='color: #718096;'>DataYad Computer Vision Course Project</p>
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
    # Header
    create_header()
    
    # Project Overview
    create_project_overview()
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Architecture Section
    create_architecture_section()
    
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