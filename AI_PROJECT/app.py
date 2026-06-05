import streamlit as st
from google import genai
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Student Report Generator",
    page_icon="🎓"
)

# -----------------------------
# Gemini Client
# -----------------------------
try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception as e:
    st.error(f"Gemini Setup Error: {e}")
    st.stop()

# -----------------------------
# PDF Function
# -----------------------------
def create_pdf(report_text):
    pdf_file = "Student_Report.pdf"

    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            report_text.replace("\n", "<br/>"),
            styles["BodyText"]
        )
    ]

    doc.build(story)

    return pdf_file

# -----------------------------
# UI
# -----------------------------
st.title("🎓 AI Student Report Generator")

st.write(
    "Enter student marks and generate an AI-powered performance report."
)

name = st.text_input("Student Name")

maths = st.number_input(
    "Maths Marks",
    min_value=0,
    max_value=100,
    value=0
)

science = st.number_input(
    "Science Marks",
    min_value=0,
    max_value=100,
    value=0
)

english = st.number_input(
    "English Marks",
    min_value=0,
    max_value=100,
    value=0
)

computer = st.number_input(
    "Computer Marks",
    min_value=0,
    max_value=100,
    value=0
)

# -----------------------------
# Generate Report
# -----------------------------
if st.button("Generate Report"):

    if not name.strip():
        st.warning("Please enter student name.")
        st.stop()

    total = maths + science + english + computer
    percentage = total / 4

    if percentage >= 90:
        grade = "A+"
    elif percentage >= 80:
        grade = "A"
    elif percentage >= 70:
        grade = "B"
    elif percentage >= 60:
        grade = "C"
    else:
        grade = "D"

    st.success("Report Generated Successfully!")

    st.subheader("📊 Academic Performance")

    st.write(f"**Student Name:** {name}")
    st.write(f"**Total Marks:** {total}/400")
    st.write(f"**Percentage:** {percentage:.2f}%")
    st.write(f"**Grade:** {grade}")

    prompt = f"""
    Student Name: {name}

    Marks:
    Maths: {maths}
    Science: {science}
    English: {english}
    Computer: {computer}

    Percentage: {percentage:.2f}%
    Grade: {grade}

    Generate:
    1. Performance Summary
    2. Strengths
    3. Areas for Improvement
    4. Study Tips

    Keep feedback positive, professional and motivational.
    """

    with st.spinner("Generating AI Feedback..."):

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            feedback = response.text

        except Exception as e:
            st.error(f"Gemini Error: {e}")
            st.stop()

    st.subheader("🤖 AI Feedback")
    st.write(feedback)

    report = f"""
    STUDENT REPORT

    Student Name: {name}

    Total Marks: {total}/400

    Percentage: {percentage:.2f}%

    Grade: {grade}

    AI Feedback:

    {feedback}
    """

    pdf_file = create_pdf(report)

    with open(pdf_file, "rb") as file:
        st.download_button(
            label="📄 Download PDF Report",
            data=file,
            file_name="Student_Report.pdf",
            mime="application/pdf"
        )
