"""
DocSage Email Tool — Send answers via Gmail SMTP.

WHY THIS FILE EXISTS:
    This tool lets the agent send the generated answer (or any content)
    via email when the user requests it. The user says something like
    "Email this answer to john@company.com" and the agent calls this tool.

    This is a DIRECT TOOL (LangChain @tool), not an MCP server.
    The agent calls it the same way it calls retriever tools or web search.

INTERVIEW POINT:
    "How does the email feature work?"
    "I built a custom LangChain tool that sends emails via Gmail SMTP.
    The agent has this tool alongside the document retrievers and web
    search. When the user asks to email an answer, the agent extracts
    the recipient address from the message and calls the send_email tool.
    The tool uses Gmail's SMTP with an App Password for authentication.
    In a larger system, this would be an MCP server so multiple agents
    could share the email capability."

    "Why a direct @tool instead of MCP?"
    "For simple, single-function tools like email, a direct @tool is
    simpler to deploy and maintain. MCP is better for complex integrations
    with multiple operations (like Jira with get_ticket, create_story,
    add_comment). The agent doesn't care — both appear as callable tools."

HOW IT WORKS:
    1. Agent decides to call send_email based on user's message.
    2. Agent extracts: recipient_email, subject, body from context.
    3. Tool connects to Gmail SMTP (smtp.gmail.com:587 with TLS).
    4. Tool sends the email using your Gmail App Password.
    5. Tool returns success/failure message to the agent.
    6. Agent relays the result to the user.
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from langchain_core.tools import tool

from docsage.config import settings

logger = logging.getLogger(__name__)


@tool
def send_email(recipient_email: str, subject: str, body: str) -> str:
    """Send an email with the given subject and body to the recipient.

    Use this tool when the user asks to email, send, or share an answer
    or conversation via email. Extract the recipient email address from
    the user's message.

    Args:
        recipient_email: The email address to send to (e.g., john@company.com).
        subject: The email subject line.
        body: The email body content (the answer or summary to send).

    Returns:
        A success or failure message.
    """
    # Read sender credentials from environment
    sender_email = os.getenv("GMAIL_ADDRESS", "")
    sender_password = os.getenv("GMAIL_APP_PASSWORD", "")

    if not sender_email or not sender_password:
        logger.error("Gmail credentials not configured in .env")
        return (
            "Email sending is not configured. The application owner needs to "
            "set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in the .env file."
        )

    # Validate recipient
    if not recipient_email or "@" not in recipient_email:
        return f"Invalid email address: '{recipient_email}'. Please provide a valid email."

    try:
        # Build the email message
        msg = MIMEMultipart("alternative")
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject

        # Create both plain text and HTML versions
        # HTML version makes the email look professional
        plain_body = body
        html_body = _format_html_email(subject, body)

        msg.attach(MIMEText(plain_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        # Connect to Gmail SMTP and send
        logger.info("Sending email to %s...", recipient_email)
        with smtplib.SMTP(settings.GMAIL_SMTP_SERVER, settings.GMAIL_SMTP_PORT) as server:
            server.starttls()  # Upgrade to secure connection
            server.login(sender_email, sender_password)
            server.send_message(msg)

        logger.info("Email sent successfully to %s", recipient_email)
        return f"Email sent successfully to {recipient_email} with subject '{subject}'."

    except smtplib.SMTPAuthenticationError:
        logger.error("Gmail authentication failed")
        return (
            "Gmail authentication failed. Please check that GMAIL_APP_PASSWORD "
            "is a valid App Password (not your regular Gmail password)."
        )
    except Exception as e:
        logger.error("Failed to send email: %s", str(e)[:100])
        return f"Failed to send email: {str(e)[:150]}"


def _format_html_email(subject: str, body: str) -> str:
    """
    Format the email body as a professional HTML email.

    Converts the plain text body into a styled HTML email with
    DocSage branding. This makes the email look polished when
    the interviewer receives it.

    Args:
        subject: The email subject.
        body: The plain text body content.

    Returns:
        HTML string for the email body.
    """
    # Convert newlines to <br> and paragraphs
    html_content = body.replace("\n\n", "</p><p>").replace("\n", "<br>")

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: #1a1a2e; color: white; padding: 15px 20px; border-radius: 8px 8px 0 0;">
            <h2 style="margin: 0;">📄 DocSage</h2>
            <p style="margin: 5px 0 0 0; font-size: 12px; opacity: 0.8;">Enterprise Document Intelligence Agent</p>
        </div>
        <div style="border: 1px solid #e0e0e0; border-top: none; padding: 20px; border-radius: 0 0 8px 8px;">
            <h3 style="color: #333;">{subject}</h3>
            <p style="color: #555; line-height: 1.6;">{html_content}</p>
        </div>
        <p style="font-size: 11px; color: #999; margin-top: 15px; text-align: center;">
            Sent by DocSage — AI-powered document intelligence
        </p>
    </body>
    </html>
    """
