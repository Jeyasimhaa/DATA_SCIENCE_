import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")


def desktop_notification(probability, risk):
    """Show a desktop popup notification. Safe to call even on servers
    without a display - failures are caught so they never crash the API."""

    try:
        from plyer import notification

        notification.notify(
            title="🚨 AI Road Safety Alert",
            message=(
                f"High Accident Risk\n\n"
                f"Probability : {probability:.2f}%\n"
                f"Risk Level : {risk}\n\n"
                f"Please Drive Carefully."
            ),
            timeout=10
        )
    except Exception as error:
        print(f"[notification] Desktop notification skipped: {error}")


def send_email(to_address, probability, risk):
    """Send an email alert. Requires EMAIL_ADDRESS / EMAIL_PASSWORD
    (an app password, not your normal Gmail password) in .env."""

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("[notification] Email skipped: EMAIL_ADDRESS/EMAIL_PASSWORD not configured in .env")
        return

    subject = "🚨 AI Road Safety Alert"
    body = (
        f"High Accident Risk Detected\n\n"
        f"Probability : {probability:.2f}%\n"
        f"Risk Level  : {risk}\n\n"
        f"Please drive carefully."
    )

    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = EMAIL_ADDRESS
    message["To"] = to_address

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, [to_address], message.as_string())
    except Exception as error:
        print(f"[notification] Email failed: {error}")


def send_sms(to_number, probability, risk):
    """Send an SMS alert via Twilio. Requires TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER in .env."""

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
        print("[notification] SMS skipped: Twilio credentials not configured in .env")
        return

    try:
        from twilio.rest import Client

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        client.messages.create(
            body=f"AI Road Safety Alert: {risk} risk ({probability:.2f}%). Drive carefully.",
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
    except Exception as error:
        print(f"[notification] SMS failed: {error}")
