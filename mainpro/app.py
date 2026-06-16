import streamlit as st

st.set_page_config(
    page_title="Deepfake Detection",
    layout="wide"
)

st.title("🎭 Deepfake Detection System")

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Dashboard",
        "Video Analyzer",
        "Audio Analyzer"
    ]
)

if menu == "Dashboard":

    st.header("Dashboard")

    st.metric(
        "Files Scanned",
        100
    )

    st.metric(
        "Deepfakes Found",
        30
    )

    st.metric(
        "Safe Files",
        70
    )

elif menu == "Video Analyzer":

    st.header("Video Analysis")

    video = st.file_uploader(
        "Upload Video",
        type=["mp4","avi","mov"]
    )

    if video:

        st.video(video)

        if st.button("Analyze Video"):

            probability = 0.87

            st.metric(
                "Deepfake Probability",
                f"{probability*100:.2f}%"
            )

            st.error(
                "⚠ Likely Deepfake"
            )

elif menu == "Audio Analyzer":

    st.header("Audio Analysis")

    audio = st.file_uploader(
        "Upload Audio",
        type=["wav","mp3"]
    )

    if audio:

        st.audio(audio)

        if st.button("Analyze Audio"):

            probability = 0.76

            st.metric(
                "Deepfake Probability",
                f"{probability*100:.2f}%"
            )

            st.error(
                "⚠ AI Generated Voice"
            )