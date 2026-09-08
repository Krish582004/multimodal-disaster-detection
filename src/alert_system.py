"""
src/alert_system.py
Broadcasts a real SOS payload via Gmail SMTP.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def dispatch_sos(event_name: str, hazard_type: str, severity: str, lat: float, lon: float) -> str:
    # ⚠️ Insert your credentials here
    SENDER_EMAIL = "disasteralert0507@gmail.com"
    APP_PASSWORD = "majm yxpv dvnd tfqp" 
    
    # For testing/demo purposes, you can send the email to yourself
    RECEIVER_EMAIL = "krishnenduchatterjee2004@gmail.com" 

    payload_text = (
        f"🚨 AUTOMATED SOS DISPATCH 🚨\n\n"
        f"HAZARD: {hazard_type.upper()}\n"
        f"LOCATION: {event_name}\n"
        f"COORDINATES: {lat}, {lon}\n"
        f"SEVERITY THREAT: {severity}\n\n"
        f"ACTION REQUIRED: Immediate local authority evaluation recommended."
    )

    if SENDER_EMAIL != "your_email@gmail.com":
        try:
            # Construct the email packet
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECEIVER_EMAIL
            msg['Subject'] = f"🚨 URGENT: {severity} Threat - {hazard_type.upper()} Detected"
            msg.attach(MIMEText(payload_text, 'plain'))

            # Connect to Gmail's secure SMTP server
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            
            # Dispatch and close
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
            server.quit()
            
            print("[SYSTEM LOG] Live Gmail SOS sent successfully!")
        except Exception as e:
            print(f"[SYSTEM LOG] Failed to send live Gmail SOS: {e}")
    else:
        print("[SYSTEM LOG] Gmail credentials missing. Simulated dispatch only.")
        
    return payload_text