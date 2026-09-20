"""Streamlit frontend for Terra Sight Multimodal API."""

from __future__ import annotations

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="ISRO Multimodal GPT", page_icon="🛰️", layout="wide")

st.title("🛰️ ISRO Earth Observation Analysis")
st.markdown("Single-image and dual-image EO analysis with layer statistics")

def render_metadata(meta: dict) -> None:
    classification = meta.get("classification", {})
    indices = meta.get("spectral_indices", {})
    layers = meta.get("layers", [])

    st.markdown("#### Classification")
    # Display classification results with confidence
    st.write(
        {
            "Label": classification.get("label", "n/a"),
            "Confidence": classification.get("confidence", "n/a"),
        }
    )

    st.markdown("#### Spectral Indices")
    st.write(indices)

    if layers:
        st.markdown("#### Layer/Band Stats")
        st.dataframe(layers, use_container_width=True)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", "http://localhost:8000")
    max_length = st.slider("Max Response Length", 32, 256, 128)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)

    st.header("About")
    st.info(

    )

single_tab, dual_tab, batch_tab = st.tabs(["Single Image", "Dual Image", "Batch"])

with single_tab:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="single")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        st.subheader("Question")
        # Input area for the user question
        question = st.text_area(
            "Enter your research question:",
            value="Classify this EO scene and summarize key spectral indicators.",
            height=110,
        )

        if st.button("Analyze Single Image", type="primary"):
            if not uploaded_file:
                st.error("Please upload an image first.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        # Prepare files and data for the API request
                        files = {
                            "image": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        }
                        data = {
                            "question": question,
                            "max_length": str(max_length),
                            "temperature": str(temperature),
                        }
                        response = requests.post(f"{api_url}/analyze", files=files, data=data, timeout=180)
                        if response.status_code == 200:
                            result = response.json()
                            st.success("Analysis complete")
                            st.markdown("### Model Response")
                            st.write(result.get("analysis", ""))
                            render_metadata(result.get("metadata", {}))
                        else:
                            st.error(f"API error {response.status_code}: {response.text}")
                    except Exception as exc:
                        st.error(f"Request failed: {exc}")

with dual_tab:
    st.subheader("Dual Image Change Analysis")
    c1, c2 = st.columns(2)

    with c1:
        image_a = st.file_uploader("Image A", type=["png", "jpg", "jpeg", "tif", "tiff"], key="dual_a")
        if image_a:
            st.image(Image.open(image_a), caption="Image A", use_container_width=True)

    with c2:
        image_b = st.file_uploader("Image B", type=["png", "jpg", "jpeg", "tif", "tiff"], key="dual_b")
        if image_b:
            st.image(Image.open(image_b), caption="Image B", use_container_width=True)

    # Dual image comparison questions
    dual_question = st.text_area(
        "Compare images:",
        value="Compare the two images and identify land-cover changes.",
        height=90,
        key="dual_q",
    )

    if st.button("Analyze Dual Images", type="primary"):
        if not image_a or not image_b:
            st.error("Please upload both images.")
        else:
            with st.spinner("Running dual-image analysis..."):
                try:
                    # Prepare multi-part files and data for dual analysis
                    files = {
                        "image_a": (image_a.name, image_a.getvalue(), image_a.type or "application/octet-stream"),
                        "image_b": (image_b.name, image_b.getvalue(), image_b.type or "application/octet-stream"),
                    }
                    data = {
                        "question": dual_question,
                        "max_length": str(max_length),
                        "temperature": str(temperature),
                    }
                    response = requests.post(f"{api_url}/analyze_dual", files=files, data=data, timeout=240)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Dual-image analysis complete")

                        st.markdown("### Change Report")
                        st.write(result.get("change_report", {}))

                        a_col, b_col = st.columns(2)
                        with a_col:
                            st.markdown("### Image A Response")
                            st.write(result.get("image_a", {}).get("analysis", ""))
                            render_metadata(result.get("image_a", {}).get("metadata", {}))

                        with b_col:
                            st.markdown("### Image B Response")
                            st.write(result.get("image_b", {}).get("analysis", ""))
                            render_metadata(result.get("image_b", {}).get("metadata", {}))
                    else:
                        st.error(f"API error {response.status_code}: {response.text}")
                except Exception as exc:
                    st.error(f"Dual request failed: {exc}")

with batch_tab:
    st.subheader("Batch Processing")
    # Allow multiple image uploads for batch processing
    batch_files = st.file_uploader(
        "Upload multiple images:",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="batch",
    )
    batch_question = st.text_input("Question for all images")

    if st.button("Process Batch"):
        if not batch_files:
            st.error("Upload at least one image for batch processing.")
        elif not batch_question.strip():
            st.error("Enter a batch question.")
        else:
            try:
                files = [
                    ("images", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in batch_files
                ]
                data = [("questions", batch_question) for _ in batch_files]
                data.extend([("max_length", str(max_length)), ("temperature", str(temperature))])
                response = requests.post(f"{api_url}/batch_analyze", files=files, data=data, timeout=300)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Batch processing complete")
                    for item in result.get("results", []):
                        with st.expander(item.get("image", "result")):
                            st.write(f"Question: {item.get('question', '')}")
                            st.write(item.get("analysis", ""))
                            render_metadata(item.get("metadata", {}))
                else:
                    st.error(f"API error {response.status_code}: {response.text}")
            except Exception as exc:
                st.error(f"Batch request failed: {exc}")
