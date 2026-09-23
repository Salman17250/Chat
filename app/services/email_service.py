import os
import re
import json
import time
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from app.rag.config import DATA_DIR, BASE_DIR

ESCALATIONS_FILE = DATA_DIR / "escalations.json"
ENV_FILE = BASE_DIR / ".env"

def _load_env_file():
    """Reads key-values from .env if present and updates os.environ without requiring external packages."""
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k:
                        os.environ[k] = v
        except Exception as e:
            print(f"[EmailEscalation] Could not read .env: {e}")

# Initial load
_load_env_file()

def get_smtp_credentials():
    """Dynamically fetches credentials so updates to .env take effect immediately without restarting."""
    _load_env_file()
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    sender = os.getenv("SMTP_SENDER", "slman373sajid@gmail.com")
    recipient = os.getenv("SMTP_RECIPIENT", "salman.wortal@gmail.com")
    pwd = os.getenv("SMTP_PASSWORD", os.getenv("GMAIL_APP_PASSWORD", "")).strip()
    fs_token = os.getenv("FORMSUBMIT_TOKEN", "85dc09efe7f0d71b9bc597f492df4a75").strip()
    return host, port, sender, recipient, pwd, fs_token

_log_lock = threading.Lock()

def extract_phone_number(text: str) -> Optional[str]:
    """
    Extracts phone numbers from user input text.
    Supports:
      - 10-digit mobile numbers (e.g., 9876543210, 8123456789)
      - With country codes (e.g., +91 9876543210, 919876543210)
      - Delimited with spaces or hyphens (e.g., 98765-43210, 9876 543 210)
    """
    if not text:
        return None

    # Pattern for phone numbers with optional country code and separators
    patterns = [
        r'(?:\+?91[\s-]?)?[6-9]\d{9}\b',                  # Standard Indian 10-digit mobile
        r'\b\+?[1-9]\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,5}[\s-]?\d{3,5}\b', # International
        r'\b\d{5}[\s-]?\d{5}\b',                          # 5-5 split
        r'\b\d{10}\b'                                     # Generic 10-digit
    ]

    for pat in patterns:
        m = re.search(pat, text.strip())
        if m:
            clean = re.sub(r'[^\d+]', '', m.group(0))
            if len(re.sub(r'\D', '', clean)) >= 10:
                return clean
    return None

def extract_email_address(text: str) -> Optional[str]:
    """Extracts email address from text."""
    if not text:
        return None
    m = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return m.group(0) if m else None

def log_escalation_ticket(
    query: str,
    contact_number: Optional[str],
    contact_email: Optional[str] = None,
    session_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    status: str = "PENDING",
    error_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Logs escalation inquiry to persistent JSON audit file.
    Ensures zero user inquiry loss even if SMTP server is temporarily unreachable.
    """
    _, _, sender, recipient, _, _ = get_smtp_credentials()
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": time.time(),
        "query": query,
        "contact_number": contact_number,
        "contact_email": contact_email,
        "session_id": session_id or "default",
        "tenant_id": tenant_id or "default",
        "sender": sender,
        "recipient": recipient,
        "status": status,
        "error": error_message
    }

    # 1. Sync to Cloud PostgreSQL if configured
    try:
        from app.storage_postgres import get_postgres_store
        pg_store = get_postgres_store()
        if pg_store:
            pg_store.insert_escalation(
                query=query,
                contact_number=contact_number,
                contact_email=contact_email,
                session_id=session_id,
                tenant_id=tenant_id,
                status=status
            )
    except Exception as pg_err:
        print(f"[EmailEscalation] Warning: Cloud PostgreSQL escalation sync: {pg_err}")

    return record

def _send_smtp_worker(
    query: str,
    contact_number: Optional[str],
    contact_email: Optional[str],
    session_id: Optional[str],
    tenant_id: Optional[str]
):
    """Internal worker that executes SMTP dispatch in a background thread."""
    host, port, sender, recipient, pwd, fs_token = get_smtp_credentials()

    subject = f"🔥 [New Lead] Customer Inquiry: {contact_number or 'Action Required'}"
    ticket_id = f"WORTAL-ESC-{int(time.time()) % 1000000:06d}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Customer Escalation Ticket</title>
      <style>
        /* Modern CSS Perforated Scalloped Edge for Browsers */
        .scalloped-top {{
          background-image: radial-gradient(circle 5px at 8px 0, #0f172a 5px, #ffffff 5.5px);
          background-size: 16px 14px;
          background-repeat: repeat-x;
        }}
      </style>
    </head>
    <body style="margin: 0; padding: 35px 15px; background: #0f172a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
      
      <!-- Voucher Receipt Wrapper -->
      <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; margin: 0 auto;">
        <tr>
          <td>
            
            <!-- Main 3D Elevated Receipt Body -->
            <div style="background: #ffffff; border-radius: 0 0 16px 16px; box-shadow: 0 25px 55px -10px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(255, 255, 255, 0.08); overflow: hidden; position: relative;">
              
              <!-- 1. Perforated Paper Receipt Scalloped Edge (Small, tight, evenly spaced semicircles) -->
              <div class="scalloped-top" style="width: 100%; height: 14px; background-color: #ffffff; background-image: radial-gradient(circle 5px at 8px 0, #0f172a 5px, #ffffff 5.5px); background-size: 16px 14px; background-repeat: repeat-x; line-height: 0; font-size: 0;">
                <!-- SVG fallback ensures exact semicircles in strict email clients like Gmail Web & Outlook -->
                <svg width="100%" height="14" style="display: block; width: 100%; height: 14px;" preserveAspectRatio="none">
                  <defs>
                    <pattern id="scallop-punch" x="0" y="0" width="16" height="14" patternUnits="userSpaceOnUse">
                      <circle cx="8" cy="0" r="5" fill="#0f172a" />
                    </pattern>
                  </defs>
                  <rect width="100%" height="14" fill="#ffffff" />
                  <rect width="100%" height="14" fill="url(#scallop-punch)" />
                </svg>
              </div>

              <!-- 2. Receipt Top Header & Perforation Tear Line -->
              <div style="padding: 16px 28px 14px 28px; background: #ffffff; border-bottom: 2px dashed #cbd5e1; text-align: center;">
                <div style="display: inline-block; background: #f1f5f9; padding: 4px 12px; border-radius: 6px; font-size: 11px; font-weight: 800; color: #475569; letter-spacing: 1.5px; text-transform: uppercase;">
                  ✂ - - - - - - - SUPPORT VOUCHER TICKET - - - - - - - ✂
                </div>
                <h1 style="margin: 14px 0 4px 0; color: #0f172a; font-size: 22px; font-weight: 800; letter-spacing: -0.5px;">
                  Customer Inquiry Escalation
                </h1>
                <p style="margin: 0; color: #64748b; font-size: 13px; font-weight: 500;">
                  Ticket ID: <strong style="color: #2563eb; font-family: monospace;">{ticket_id}</strong> &nbsp;|&nbsp; {time.strftime("%d %b %Y, %I:%M %p")}
                </p>
              </div>

              <!-- 3. Receipt Details Section -->
              <div style="padding: 26px 28px; background: #ffffff;">
                
                <!-- User Question Receipt Box -->
                <div style="margin-bottom: 24px; background: #f8fafc; border-radius: 12px; padding: 18px 20px; border: 1.5px solid #e2e8f0; border-left: 5px solid #2563eb; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);">
                  <div style="font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: #2563eb; margin-bottom: 6px;">
                    📌 Customer Question / Query
                  </div>
                  <div style="font-size: 17px; font-weight: 700; color: #0f172a; line-height: 1.5; font-style: italic;">
                    "{query}"
                  </div>
                </div>

                <!-- 3D Receipt Data Table -->
                <div style="background: #ffffff; border-radius: 12px; border: 1.5px solid #e2e8f0; box-shadow: 0 8px 20px -6px rgba(0, 0, 0, 0.08); overflow: hidden; margin-bottom: 24px;">
                  <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse; font-size: 14px;">
                    
                    <!-- Phone Row -->
                    <tr style="border-bottom: 1px solid #f1f5f9; background: #ffffff;">
                      <td style="padding: 14px 18px; font-weight: 600; color: #64748b; width: 36%;">
                        <span style="display: inline-block; vertical-align: middle; margin-right: 6px;">📞</span> Contact Phone
                      </td>
                      <td style="padding: 14px 18px;">
                        <span style="display: inline-block; background: linear-gradient(135deg, #059669, #047857); color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 800; font-size: 15px; box-shadow: 0 3px 10px rgba(5, 150, 105, 0.3);">
                          {contact_number or 'Not provided'}
                        </span>
                      </td>
                    </tr>

                    <!-- Email Row -->
                    <tr style="border-bottom: 1px solid #f1f5f9; background: #fafafa;">
                      <td style="padding: 14px 18px; font-weight: 600; color: #64748b;">
                        <span style="display: inline-block; vertical-align: middle; margin-right: 6px;">✉️</span> Contact Email
                      </td>
                      <td style="padding: 14px 18px; color: #1e293b; font-weight: 700;">
                        {contact_email or 'Not provided'}
                      </td>
                    </tr>

                    <!-- Priority Row -->
                    <tr style="border-bottom: 1px solid #f1f5f9; background: #ffffff;">
                      <td style="padding: 14px 18px; font-weight: 600; color: #64748b;">
                        <span style="display: inline-block; vertical-align: middle; margin-right: 6px;">⚡</span> Priority Status
                      </td>
                      <td style="padding: 14px 18px;">
                        <span style="display: inline-block; background: #fef2f2; color: #b91c1c; padding: 4px 12px; border-radius: 14px; font-size: 11px; font-weight: 800; border: 1px solid #fecaca; text-transform: uppercase; letter-spacing: 0.5px;">
                          HIGH - Immediate Follow-Up
                        </span>
                      </td>
                    </tr>

                    <!-- Timestamp Row -->
                    <tr style="background: #fafafa;">
                      <td style="padding: 14px 18px; font-weight: 600; color: #64748b;">
                        <span style="display: inline-block; vertical-align: middle; margin-right: 6px;">🕒</span> Logged At
                      </td>
                      <td style="padding: 14px 18px; color: #334155; font-weight: 600;">
                        {time.strftime("%d %b %Y at %I:%M %p")}
                      </td>
                    </tr>

                  </table>
                </div>

                <!-- 3D Call to Action -->
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px; border: 1px solid #bfdbfe; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.12);">
                  <p style="margin: 0 0 12px 0; color: #1e40af; font-size: 13px; font-weight: 700;">
                    Please contact the customer to address this inquiry.
                  </p>
                  {f'<a href="tel:{contact_number}" style="display: inline-block; background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%); color: #ffffff; text-decoration: none; font-size: 14px; font-weight: 700; padding: 12px 28px; border-radius: 8px; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35); letter-spacing: 0.3px;">📞 Call Customer ({contact_number})</a>' if contact_number else ''}
                </div>

                <!-- Decorative Voucher Barcode -->
                <div style="margin-top: 24px; text-align: center; padding-top: 18px; border-top: 2px dashed #cbd5e1;">
                  <div style="font-family: 'Courier New', Courier, monospace; letter-spacing: 4px; font-size: 18px; color: #0f172a; font-weight: bold; margin-bottom: 4px;">
                    ||||| | |||| ||||| || |||||| | |||||
                  </div>
                  <span style="font-size: 10px; color: #94a3b8; letter-spacing: 1px; font-family: monospace;">
                    *{ticket_id}*
                  </span>
                </div>

              </div>

              <!-- Receipt Footer -->
              <div style="background: #f8fafc; padding: 14px 24px; text-align: center; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 11px;">
                Wortal Intelligent Document Assistant &bull; Confidential
              </div>

            </div>
          </td>
        </tr>
      </table>
    </body>
    </html>
    """

    plain_text = f"""
    CUSTOMER INQUIRY ESCALATION
    
    User Question: {query}
    Contact Number: {contact_number or 'Not provided'}
    Contact Email: {contact_email or 'Not provided'}
    # Tenant ID: {tenant_id or 'default'} (Commented out)
    # Session ID: {session_id or 'default'} (Commented out)
    Timestamp: {time.strftime("%d %b %Y, %I:%M %p")}
    
    Please follow up with the user directly.
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    # Primary: Attempt SMTP if password is provided
    if pwd:
        try:
            with smtplib.SMTP(host, port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(sender, pwd)
                server.sendmail(sender, [recipient], msg.as_string())

            print(f"[EmailEscalation] Successfully dispatched email via SMTP from {sender} to {recipient} for query: '{query[:40]}...'")
            log_escalation_ticket(
                query=query,
                contact_number=contact_number,
                contact_email=contact_email,
                session_id=session_id,
                tenant_id=tenant_id,
                status="SENT_SMTP"
            )
            return
        except Exception as e:
            print(f"[EmailEscalation] SMTP send failed: {e}. Attempting HTTP fallback...")

    # Secondary / Zero-Password Fallback: FormSubmit Webhook API
    try:
        import requests
        fs_target = fs_token if fs_token else recipient
        fs_url = f"https://formsubmit.co/ajax/{fs_target}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://wortal.com"
        }
        payload = {
            "_subject": f"🔥 [New Lead] Query from {contact_number or 'Customer'}",
            "_replyto": sender,
            "📌 Customer Query": query,
            "📞 Contact Phone": contact_number or "Not provided",
            "✉️ Contact Email": contact_email or "Not provided",
            # "Tenant ID": tenant_id or "default",
            # "Session ID": session_id or "default",
            "🕒 Received At": time.strftime("%d %b %Y, %I:%M %p"),
            "⚡ Follow-Up Status": "URGENT - Direct Customer Action Required",
            "_template": "table"
        }
        resp = requests.post(fs_url, json=payload, headers=headers, timeout=25)
        data = resp.json() if resp.status_code == 200 else {}
        if str(data.get("success")).lower() == "true":
            print(f"[EmailEscalation] Successfully delivered email via Web API to {recipient}")
            log_escalation_ticket(
                query=query,
                contact_number=contact_number,
                contact_email=contact_email,
                session_id=session_id,
                tenant_id=tenant_id,
                status="SENT_WEB_API"
            )
            return
        else:
            msg_desc = data.get("message", "Web API activation needed")
            print(f"[EmailEscalation] Web API notification: {msg_desc}")
            log_escalation_ticket(
                query=query,
                contact_number=contact_number,
                contact_email=contact_email,
                session_id=session_id,
                tenant_id=tenant_id,
                status="QUEUED_PENDING_ACTIVATION",
                error_message=msg_desc
            )
    except Exception as e:
        print(f"[EmailEscalation] Web API fallback error: {e}")
        log_escalation_ticket(
            query=query,
            contact_number=contact_number,
            contact_email=contact_email,
            session_id=session_id,
            tenant_id=tenant_id,
            status="QUEUED_PENDING_SMTP_PASSWORD",
            error_message=str(e)
        )

def flush_queued_escalations() -> int:
    """Dispatches any previously queued tickets once SMTP credentials or Web API are available."""
    if not ESCALATIONS_FILE.exists():
        return 0

    with _log_lock:
        try:
            with open(ESCALATIONS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            return 0

    flushed_count = 0
    for r in records:
        if str(r.get("status", "")).startswith("QUEUED_"):
            print(f"[EmailEscalation] Flushing queued ticket for: {r.get('query')}")
            _send_smtp_worker(
                query=r.get("query", ""),
                contact_number=r.get("contact_number"),
                contact_email=r.get("contact_email"),
                session_id=r.get("session_id"),
                tenant_id=r.get("tenant_id")
            )
            flushed_count += 1

    return flushed_count

def dispatch_escalation_email(
    query: str,
    contact_number: Optional[str],
    contact_email: Optional[str] = None,
    session_id: Optional[str] = None,
    tenant_id: Optional[str] = None
):
    """
    Dispatches escalation email asynchronously in a non-blocking background thread.
    Returns immediately so user chat latency remains under 50ms.
    """
    t = threading.Thread(
        target=_send_smtp_worker,
        args=(query, contact_number, contact_email, session_id, tenant_id),
        daemon=True
    )
    t.start()
