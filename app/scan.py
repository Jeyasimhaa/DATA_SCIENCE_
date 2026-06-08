import streamlit as st
import cv2
import numpy as np
from PIL import Image
from urllib.parse import urlparse

# Page Config
st.set_page_config(
    page_title="QR Code Safety Checker",
    page_icon="🔒",
    layout="centered"
)

st.title("🔒 QR Code Safety Checker")
st.write("Upload a QR Code image to check whether the URL is safe or unsafe.")

# Upload QR Image
uploaded_file = st.file_uploader(
    "Upload QR Code Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(image, caption="Uploaded QR Code", use_container_width=True)

    # Convert PIL image to OpenCV format
    img = np.array(image)

    # Convert RGB to BGR (OpenCV format)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detect and Decode QR
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)

    st.markdown("---")

    if data:

        st.subheader("📄 Decoded Content")
        st.code(data)

        parsed = urlparse(data)

        st.subheader("🛡️ Safety Analysis")

        if parsed.scheme == "https":
            st.success("✅ SAFE")
            st.write("HTTPS detected. Secure connection.")

        elif parsed.scheme == "http":
            st.error("⚠️ UNSAFE")
            st.write("HTTP detected. Website is not using encryption.")

        else:
            st.warning("❓ UNKNOWN")
            st.write("The QR code does not contain a valid website URL.")

        # Additional Information
        if parsed.netloc:
            st.subheader("🌐 URL Details")
            st.write(f"**Domain:** {parsed.netloc}")
            st.write(f"**Protocol:** {parsed.scheme}")

    else:
        st.error("❌ No QR Code detected in the uploaded image.")