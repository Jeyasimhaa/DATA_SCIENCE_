import streamlit as st
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import base64

# -----------------------------
# OpenAI Client
# -----------------------------
client = OpenAI(
    api_key="AQ.Ab8RN6KAKgIPfMnrDg0HEttqKkQQN7ge7f8qR-GjclNJs9P4WQ"
)

# -----------------------------
# PDF Generator Function
# -----------------------------
def create_pdf(story_text):

    pdf_file = "AI_Story.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    content = [
        Paragraph("AI Story Creator", styles['Title']),
        Paragraph(story_text, styles['BodyText'])
    ]

    doc.build(content)

    return pdf_file


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="AI Story Creator",
    page_icon="📖"
)

st.title("📖 AI Story Creator")

st.write(
    "Generate Stories, Characters, Images and Download as PDF"
)

prompt = st.text_area(
    "Enter Story Prompt",
    "Write a short adventure story for kids about a robot exploring Mars."
)

if st.button("Generate Story"):

    # -----------------------------
    # Story Generation
    # -----------------------------
    with st.spinner("Generating Story..."):

        story_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Create:

                    1. Story Title
                    2. Main Characters
                    3. Short Children's Story

                    Prompt:
                    {prompt}
                    """
                }
            ]
        )

        story = story_response.choices[0].message.content

    st.success("Story Generated Successfully!")

    st.subheader("📚 Generated Story")
    st.write(story)

    # -----------------------------
    # Image Generation
    # -----------------------------
    with st.spinner("Generating Story Image..."):

        image_response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024"
        )

        image_base64 = image_response.data[0].b64_json

        image_bytes = base64.b64decode(image_base64)

        st.subheader("🎨 Story Illustration")
        st.image(
            image_bytes,
            caption="AI Generated Story Image"
        )

    # -----------------------------
    # PDF Export
    # -----------------------------
    pdf_file = create_pdf(story)

    with open(pdf_file, "rb") as file:

        st.download_button(
            label="📄 Download Story PDF",
            data=file,
            file_name="AI_Story.pdf",
            mime="application/pdf"
        )