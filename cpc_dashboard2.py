#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Cleaning Profile Classification (CPC)", layout="wide")



# Sidebar footer note
st.sidebar.markdown("---")
st.sidebar.markdown("**Guided by DR G.R.Brindha**")
st.sidebar.markdown("**Pavithra G**")

# Add pages
page = st.sidebar.radio("Go to", ["Overview", "Model Dashboard"])

if page == "Overview":
    # ---------- OVERVIEW PAGE ----------
    # Logo and title
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image("images/logo.jpg", width=100)
    with col2:
        st.title("Cleaning Profile Classification using Convolutional Neural Network in Stencil Printing")

    st.markdown("### Research Overview")
    st.markdown("""
    This research proposes a novel framework to classify the **stencil cleaning profile** in the stencil printing process (SPP) on a real-time basis.
    The stencil cleaning operation is necessary to reduce printing defects. Proper control of the cleaning profile selection process determines the quality and efficiency of SPP.

    - **Wet profile**: High-quality cleaning, but higher time and material usage.
    - **Dry profile**: More efficient, but may leave residues.

    The research aims to build an **intelligent model** to guide this decision-making using:
    - Historical quality measures
    - Time-series data encoded into **images**
    - Classification using **CNN with transfer learning**

    The model improves the stencil cleaning process by:
    - Enhancing defect detection
    - Improving SPP productivity and quality

    Experimental results show:
    - **CPC CNN model** outperforms other architectures.
    """)

elif page == "Model Dashboard":
    # ---------- MODEL METRICS PAGE ----------
    st.title("Model Metrics Dashboard")

    # Sample data
    data = {
        "RP": {"Accuracy": 0.85, "Precision": {"WET": 0.53, "DRY": 0.95}, "Recall": {"WET": 0.77, "DRY": 0.87}, "Parameters": 7495905},
        "GADF": {"Accuracy": 0.92, "Precision": {"WET": 0.82, "DRY": 0.94}, "Recall": {"WET": 0.69, "DRY": 0.97}, "Parameters": 7395905},
        "GASF": {"Accuracy": 0.93, "Precision": {"WET": 0.97, "DRY": 0.93}, "Recall": {"WET": 0.62, "DRY": 0.97}, "Parameters": 7395905},
        "VGG16Net": {"Train Accuracy": 0.88, "Test Accuracy": 0.89},
        "VGGNet19": {"Train Accuracy": 0.84, "Test Accuracy": 0.90},
        "Mobilenet": {"Train Accuracy": 0.71, "Test Accuracy": 0.52},
        "AlexNet": {"Test Accuracy": 0.90},
        "LENET": {"Accuracy": 0.96, "Parameters": 389646},
        "LENET WITH SMOTE": {"Accuracy": 0.95, "Parameters": 389646},
        "LENET WITH PCA, SMOTE": {"Accuracy": 0.96, "Parameters": 13314},
        "LENET WITH PCA": {"Accuracy": 0.93, "Parameters": 15234},
        "LIME-LENET": {"Accuracy": 0.86, "Parameters": 389646},
        "LIME-LENET WITH PCA, SMOTE": {"Accuracy": 0.85, "Parameters": 16930},
        "LIME-LENET WITH PCA": {"Accuracy": 0.85, "Parameters": 16900},
        "LIME-LENET WITH SMOTE": {"Accuracy": 0.86, "Parameters": 1012520},
        "SHALLOW CPC": {"Accuracy": 0.91, "Parameters": 7396481},
        "SHALLOW CPC WITH SMOTE": {"Accuracy": 0.99, "Parameters": 7396481},
        "SHALLOW CPC WITH PCA, SMOTE": {"Accuracy": 0.96, "Parameters": 11521},
        "SHALLOW CPC WITH PCA": {"Accuracy": 0.98, "Parameters": 16257},
        "LIME - SHALLOW WITH SMOTE": {"Accuracy": 0.90, "Parameters": 3945025},
        "LIME - SHALLOW WITH SMOTE , PCA": {"Accuracy": 0.72, "Parameters": 12865},
        "LIME- SHALLOW WITH PCA": {"Accuracy": 0.74, "Parameters": 12865},
        "LIME SHALLOW CPC": {"Accuracy": 0.93, "Parameters": 3945025},
    }

    # Top logo
    st.image("images/logo.png", width=100)

    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox("Select a Model to View Metrics", list(data.keys()))

        st.markdown(f"### 📊 Performance of {selected_model}")
        metrics = data[selected_model]
        for key, value in metrics.items():
            if isinstance(value, dict):
                st.write(f"**{key}**:")
                for subkey, subvalue in value.items():
                    st.write(f"- {subkey}: {subvalue}")
            elif key != "Parameters":
                st.write(f"**{key}**: {value}")

    with col2:
        param_model = st.selectbox("Select a Model to View Parameters", list(data.keys()))
        if "Parameters" in data[param_model]:
            st.success(f"🔧 Parameters: {data[param_model]['Parameters']:,}")
        else:
            st.warning("No parameter data for this model.")

    st.markdown("---")
    st.markdown("### 🖼️ Image Visualizations")
    img1, img2, img3, img4 = st.columns(4)
    with img1:
        if st.button("Normal DRY"):
            st.image("images/dry_0.png", caption="Normal DRY")
    with img2:
        if st.button("Normal WET"):
            st.image("images/wet_0.png", caption="Normal WET")
    with img3:
        if st.button("LIME DRY"):
            st.image("images/5.png", caption="LIME DRY")
    with img4:
        if st.button("LIME WET"):
            st.image("images/149.png", caption="LIME WET")

    st.markdown("---")
    st.markdown("### 📈 Compare Accuracy Between Models")
    compare_models = st.multiselect("Select Models to Compare", list(data.keys()))

    if compare_models:
        acc_values = []
        labels = []
        for model in compare_models:
            acc = None
            if "Accuracy" in data[model]:
                acc = data[model]["Accuracy"]
            elif "Test Accuracy" in data[model]:
                acc = data[model]["Test Accuracy"]
            if acc is not None:
                acc_values.append(acc)
                labels.append(model)

        if acc_values:
            fig, ax = plt.subplots(figsize=(10, len(labels) * 0.5))
            ax.barh(labels, acc_values, color="skyblue")
            ax.set_xlabel("Accuracy")
            ax.set_title("Model Accuracy Comparison")
            st.pyplot(fig)
        else:
            st.warning("No accuracy data available for selected models.")

