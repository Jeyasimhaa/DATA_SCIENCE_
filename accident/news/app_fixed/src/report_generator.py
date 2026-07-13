import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "..", "output", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_report(
    weather,
    speed,
    road,
    traffic,
    time_of_day,
    latitude,
    longitude,
    probability,
    risk,
    recommendation
):

    filename = os.path.join(
        REPORTS_DIR,
        datetime.now().strftime("Accident_Report_%Y%m%d_%H%M%S.pdf")
    )

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph(
        "<b><font size=18>AI Road Accident Prediction Report</font></b>",
        styles["Title"]
    )
    elements.append(title)

    elements.append(
        Paragraph(f"Generated: {datetime.now()}", styles["Normal"])
    )

    data = [
        ["Weather", weather],
        ["Speed", f"{speed} km/h"],
        ["Road Condition", road],
        ["Traffic", traffic],
        ["Time", time_of_day],
        ["Latitude", latitude],
        ["Longitude", longitude],
        ["Probability", f"{probability:.2f}%"],
        ["Risk Level", risk],
        ["Recommendation", recommendation]
    ]

    table = Table(data, colWidths=[170, 250])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
    ]))

    elements.append(table)
    doc.build(elements)

    return filename
