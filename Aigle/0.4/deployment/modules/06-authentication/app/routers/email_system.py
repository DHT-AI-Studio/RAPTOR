import smtplib
from email.message import EmailMessage
from enum import Enum

from fastapi import APIRouter, Form, HTTPException

from app.core.config import settings

router = APIRouter(tags=["Public"])

ContactSubject = Enum(
    "ContactSubject",
    {
        "General Inquiry":        "General Inquiry",
        "Technical Support":      "Technical Support",
        "Business Collaboration": "Business Collaboration",
        "Feedback":               "Feedback",
        "Other":                  "Other",
    },
)


def _send(to: str, subject: str, body: str):
    if not settings.SMTP_HOST:
        raise HTTPException(status_code=503, detail="SMTP not configured")
    msg = EmailMessage()
    msg["From"] = settings.SMTP_FROM or settings.SMTP_USER
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")


@router.post("/contact-dhtsolution")
async def contact_form(
    name: str = Form(...),
    email: str = Form(...),
    subject: ContactSubject = Form(...),
    message: str = Form(...),
):
    """
    聯絡 DHT Solution 表單。訊息寄送至公司信箱（SMTP_TO_EMAIL）。
    不需要 token。
    """
    if not settings.SMTP_TO_EMAIL:
        raise HTTPException(status_code=503, detail="Contact recipient not configured (SMTP_TO_EMAIL)")
    _send(
        to=settings.SMTP_TO_EMAIL,
        subject=f"Contact Form: {subject.value}",
        body=f"From: {name} <{email}>\n\n{message}",
    )
    return {"message": "Your message has been received by DHT Solution. We will review it and get back to you as soon as possible."}


@router.post("/send-email")
async def send_email(
    name: str = Form(...),
    email: str = Form(...),
    subject: str = Form(...),
    message: str = Form(...),
):
    """
    由系統寄送 email 給指定收件者。適用於通知類情境。
    不需要 token。
    """
    _send(
        to=email,
        subject=f"Message from DHT Solution: {subject}",
        body=f"Hello {name},\n\n{message}",
    )
    return {"message": "Your message has been successfully sent."}
