import streamlit as st
from PIL import Image
import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from efficientnet_pytorch import EfficientNet

# Set page configuration
st.set_page_config(
    page_title="Alzheimer's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4257B2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5C7AEA;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .normal-result {
        background-color: #D4EDDA;
        color: #155724;
    }
    .mild-result {
        background-color: #FFF3CD;
        color: #856404;
    }
    .moderate-result {
        background-color: #FFE5D9;
        color: #7D4E57;
    }
    .severe-result {
        background-color: #F8D7DA;
        color: #721C24;
    }
    .stProgress > div > div > div > div {
        background-color: #4257B2;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain--v2.png", width=100)
    st.markdown("<h1 style='text-align: center;'>Alzheimer's Detection</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Navigation")
    page = st.radio("", ["Home", "Upload & Predict", "About"])

    st.markdown("---")
    st.markdown("### About the Model")
    st.markdown("""
    This application uses an EfficientNet-B0 model trained on MRI scans to detect Alzheimer's disease stages:
    - Non-demented
    - Very mild Alzheimer's
    - Mild Alzheimer's
    - Moderate Alzheimer's
    """)

    st.markdown("---")
    st.markdown("### Dataset Distribution")
    try:
        st.image("Visualizations/class_distribution.png", caption="Class Distribution", use_column_width=True)
    except:
        st.info("Class distribution visualization not available.")

# Load the model which is in Src/alzheimer_efficientnet_model.pth
MODEL_PATH = os.path.join('Src', 'alzheimer_efficientnet_model.pth')

# Preprocess image
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Predict
def predict(image, model):
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
    return predicted.item(), probabilities.numpy()

# Function to create a bar chart for prediction probabilities
def create_prediction_chart(probabilities, labels):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']

    # Create horizontal bar chart
    bars = ax.barh(labels, probabilities, color=colors)

    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                va='center', fontsize=10)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig

# Home page
if page == "Home":
    st.markdown("<h1 class='main-header'>🧠 Alzheimer's Disease Detection</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Welcome to the Alzheimer's Detection Tool</h2>", unsafe_allow_html=True)
        st.markdown("""
        This application uses deep learning to analyze MRI scans and detect signs of Alzheimer's disease.

        **How it works:**
        1. Upload an MRI scan image
        2. Our AI model analyzes the image
        3. Get instant results showing the likelihood of different Alzheimer's stages

        Early detection of Alzheimer's disease is crucial for effective treatment and care planning.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h3 class='sub-header'>Key Features</h3>", unsafe_allow_html=True)
        col1a, col1b, col1c = st.columns(3)

        with col1a:
            st.markdown("#### 🔍 Advanced Analysis")
            st.markdown("Powered by EfficientNet deep learning model")

        with col1b:
            st.markdown("#### ⚡ Fast Results")
            st.markdown("Get predictions in seconds")

        with col1c:
            st.markdown("#### 📊 Detailed Insights")
            st.markdown("View probability breakdown for each class")

    with col2:
        st.image("https://img.icons8.com/color/240/000000/brain-scan.png", use_column_width=True)
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.button("Go to Prediction Tool", on_click=lambda: st.session_state.update({"page": "Upload & Predict"}))
        st.markdown("</div>", unsafe_allow_html=True)

# Upload & Predict page
elif page == "Upload & Predict":
    st.markdown("<h1 class='main-header'>MRI Analysis</h1>", unsafe_allow_html=True)

    # Load model
    model_load_state = st.info("Loading model...")
    try:
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        model_loaded = True
        model_load_state.success("Model loaded successfully!")
    except FileNotFoundError:
        model_load_state.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
        model_loaded = False

    st.markdown("<h3 class='sub-header'>Upload an MRI Scan</h3>", unsafe_allow_html=True)

    # Create two columns for upload and preview
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Select Image")
        uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

        # Sample images option
        st.markdown("#### Or try a sample image")
        sample_dir = "train"
        if os.path.exists(sample_dir):
            sample_categories = ["No Impairment", "Very Mild Impairment", "Mild_Impairment", "Moderate Impairment"]
            selected_category = st.selectbox("Select category", sample_categories)

            # Find sample images in the selected category
            category_path = os.path.join(sample_dir, selected_category)
            if os.path.exists(category_path):
                sample_images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if sample_images:
                    selected_sample = st.selectbox("Select a sample image", sample_images)
                    if st.button("Use this sample"):
                        sample_path = os.path.join(category_path, selected_sample)
                        with open(sample_path, "rb") as f:
                            uploaded_file = f

    # Display and analyze the image
    if uploaded_file is not None and model_loaded:
        with col2:
            st.markdown("#### Image Preview")
            image = Image.open(uploaded_file).convert('RGB')  # Ensure image is RGB
            st.image(image, caption='Uploaded MRI Scan', use_column_width=True)

        st.markdown("<h3 class='sub-header'>Analysis Results</h3>", unsafe_allow_html=True)

        # Show a progress bar for analysis
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate processing steps
        status_text.text("Preprocessing image...")
        progress_bar.progress(25)
        time.sleep(0.5)

        preprocessed_image = preprocess(image)

        status_text.text("Running model inference...")
        progress_bar.progress(50)
        time.sleep(0.5)

        label, probabilities = predict(preprocessed_image, model)

        status_text.text("Analyzing results...")
        progress_bar.progress(75)
        time.sleep(0.5)

        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)

        # Clear the progress indicators
        status_text.empty()

        # Display results
        labels = ['Mild Alzheimer\'s', 'Moderate Alzheimer\'s', 'Non-demented', 'Very Mild Alzheimer\'s']

        if label in range(len(labels)):
            result_label = labels[label]

            # Determine result class for styling
            if label == 2:  # Non-demented
                result_class = "normal-result"
            elif label == 3:  # Very Mild
                result_class = "mild-result"
            elif label == 0:  # Mild
                result_class = "moderate-result"
            else:  # Moderate
                result_class = "severe-result"

            # Display the prediction result
            st.markdown(f"<div class='result-box {result_class}'><h2>Prediction: {result_label}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p>The model predicts that this MRI scan shows signs of <strong>{result_label}</strong>.</p></div>", unsafe_allow_html=True)

            # Create two columns for the chart and explanation
            col1, col2 = st.columns([3, 2])

            with col1:
                # Create and display probability chart
                fig = create_prediction_chart(probabilities, labels)
                st.pyplot(fig)

            with col2:
                st.markdown("### Understanding the Results")
                st.markdown("""
                - **Non-demented**: No signs of Alzheimer's disease
                - **Very Mild Alzheimer's**: Early signs of cognitive decline
                - **Mild Alzheimer's**: Noticeable memory problems and cognitive difficulties
                - **Moderate Alzheimer's**: Significant memory loss and difficulty with daily activities
                """)

                st.markdown("### Next Steps")
                st.markdown("""
                Remember that this tool is for informational purposes only. 
                For a proper diagnosis, please consult with a healthcare professional.
                """)
        else:
            st.error('An error occurred during classification.')

# About page
elif page == "About":
    st.markdown("<h1 class='main-header'>About This Project</h1>", unsafe_allow_html=True)

    st.markdown("""
    ## Project Overview

    This application is designed to assist in the early detection of Alzheimer's disease using MRI scans. 
    Leveraging deep learning techniques, specifically EfficientNet architecture, the model has been trained 
    to identify patterns in brain MRI scans that are indicative of different stages of Alzheimer's disease.

    ## How It Works

    1. **Image Upload**: Users upload an MRI scan image
    2. **Preprocessing**: The image is resized, normalized, and prepared for the model
    3. **Model Inference**: Our trained EfficientNet model analyzes the image
    4. **Result Visualization**: Results are displayed with probability scores for each class

    ## Technical Details

    - **Model**: EfficientNet-B0
    - **Training Data**: Dataset of brain MRI scans with four classes
    - **Accuracy**: The model achieves high accuracy in distinguishing between different stages of Alzheimer's

    ## Disclaimer

    This tool is intended for educational and research purposes only. It should not be used as a substitute 
    for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or 
    other qualified health provider with any questions you may have regarding a medical condition.

    ## Credits

    - MRI dataset from Kaggle
    - Built with Streamlit and PyTorch
    """)

    # Contact information
    st.markdown("---")
    st.markdown("### Contact")
    st.markdown("For questions or feedback about this project, please open an issue on the GitHub repository.")

    # GitHub link
    st.markdown("### Source Code")
    st.markdown("The source code for this project is available on GitHub.")
