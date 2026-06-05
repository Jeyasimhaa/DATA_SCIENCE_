import streamlit as st
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------------
# Streamlit Config
# ----------------------------------

st.set_page_config(
    page_title="AI Student Report Generator",
    page_icon="🎓"
)

# ----------------------------------
# Gemini Configuration
# ----------------------------------

try:
    api_key = st.secrets["GEMINI_API_KEY"]

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")

except Exception as e:
    st.error(f"Gemini Setup Error: {e}")
    st.stop()

# ----------------------------------
# PDF Generator
# ----------------------------------

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

# ----------------------------------
# UI
# ----------------------------------

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

# ----------------------------------
# Generate Report
# ----------------------------------

if st.button("Generate Report"):

    if not name.strip():
        st.warning("Please enter a student name.")
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

    Keep the feedback positive,
    professional and motivational.
    """

    with st.spinner("Generating AI Feedback..."):

        try:

            response = model.generate_content(prompt)

            feedback = response.text

        except Exception as e:

            st.error("Gemini API Error")
            st.code(str(e))

            st.write("Secret Loaded:", "GEMINI_API_KEY" in st.secrets)

            if "GEMINI_API_KEY" in st.secrets:
                st.write(
                    "Key Prefix:",
                    st.secrets["GEMINI_API_KEY"][:10]
                )

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
