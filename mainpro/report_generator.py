from reportlab.pdfgen import canvas

def generate_report(
    filename,
    prediction,
    probability
):

    pdf = canvas.Canvas(
        f"reports/{filename}.pdf"
    )

    pdf.drawString(
        100,
        800,
        "Deepfake Detection Report"
    )

    pdf.drawString(
        100,
        760,
        f"File: {filename}"
    )

    pdf.drawString(
        100,
        730,
        f"Prediction: {prediction}"
    )

    pdf.drawString(
        100,
        700,
        f"Probability: {probability}"
    )

    pdf.save()