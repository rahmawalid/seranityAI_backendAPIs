import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

def send_verification_email(email, token):
    link = f"http://localhost:5001/verify-email/{token}"

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; padding: 20px;">
        <h2>Welcome to SeranityAI 👋</h2>
        <p>Thank you for registering. To verify your email address, please click the button below:</p>
        <p style="margin: 30px 0;">
          <a href="{link}" style="
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            text-decoration: none;
            border-radius: 6px;
            display: inline-block;
          ">Verify Email</a>
        </p>
        <p>If the button doesn’t work, copy and paste the link into your browser:</p>
        <p><a href="{link}">{link}</a></p>
        <br>
        <p style="font-size: 12px; color: #888;">This link will expire in 1 hour.</p>
      </body>
    </html>
    """

    msg = MIMEText(html_content, "html")
    msg['Subject'] = "Verify Your Email - SeranityAI"
    msg['From'] = os.environ.get("EMAIL_USER")
    msg['To'] = email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(os.environ.get("EMAIL_USER"), os.environ.get("EMAIL_PASS"))
            server.send_message(msg)
            print(f"✅ Verification email sent to {email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")