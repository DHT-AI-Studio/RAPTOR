# app/main.py
from fastapi import APIRouter, Form, HTTPException
import smtplib
from email.message import EmailMessage
import json 
from pathlib import Path
from enum import Enum
# 讀取 JSON 文件
JSON_FILE_PATH = Path(__file__).parent.parent / "config" / "email.json"

with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    JSON_CONFIG = json.load(f)

# SMTP settings
SMTP_HOST = JSON_CONFIG["SMTP"]["SMTP_HOST"]            # SMTP server
SMTP_PORT = JSON_CONFIG["SMTP"]["SMTP_PORT"]            # SMTP port (TLS)
SMTP_USER = JSON_CONFIG["SMTP"]["SMTP_USER"]            # Sender email
SMTP_PASSWORD = JSON_CONFIG["SMTP"]["SMTP_PASSWORD"]    # Email password or app-specific password
TO_EMAIL = JSON_CONFIG["SMTP"]["TO_EMAIL"]              # The email address where you want to receive messages.

# 動態生成 Enum（value = 顯示文字）
SubjectEnum = Enum("SubjectEnum", JSON_CONFIG["email_subject"])

router = APIRouter()


@router.post("/contact-dhtsolution")
async def contact_form(
    name: str = Form(...),
    email: str = Form(...),
    subject: SubjectEnum = Form(...),
    message: str = Form(...)
):
    try:
        # 建立郵件內容
        msg = EmailMessage()
        msg["From"] = SMTP_USER
        msg["To"] = TO_EMAIL
        msg["Subject"] = f"Contact Form: {subject.value}"
        msg.set_content(f"From: {name} <{email}>\n\n{message}")

        # 連線 SMTP 寄信
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        return {"message": f"Your message has been received by dhtslution. We will review it and get back to you as soon as possible."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@router.post("/send-email")
async def contact_form(
    name: str = Form(...),
    email: str = Form(...),
    subject: str = Form(...),
    message: str = Form(...)
):
    try:
        # 建立郵件內容
        msg = EmailMessage()
        msg["From"] = SMTP_USER
        msg["To"] = email
        msg["Subject"] = f"Send form dhtsolution: {subject}"
        msg.set_content(f"Hello: {name}\n\n{message}")

        # 連線 SMTP 寄信
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        return {"message": f"Your message has been successfully sent."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

# @router.post("/notifications")
# async def notifications(
#     name: str = Form(...),
#     email: str = Form(...),
#     subject: str = Form(...),
#     message: str = Form(...)
# ):
#     """
#     Account information update notifications in Keycloak
#     """
#     try:
#         # 建立郵件內容
#         msg = EmailMessage()
#         msg["From"] = SMTP_USER
#         msg["To"] = email
#         msg["Subject"] = f"Send form dhtsolution: {subject}"
#         msg.set_content(f"Hello: {name}\n\n{message}")

#         # 連線 SMTP 寄信
#         with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
#             server.starttls()
#             server.login(SMTP_USER, SMTP_PASSWORD)
#             server.send_message(msg)

#         return {"message": f"Your message has been successfully sent."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")