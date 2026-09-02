from twilio.rest import Client
import os

# To make this live, you will get free credentials from twilio.com
# For now, we leave them as placeholders so the app doesn't crash
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "your_account_sid_here")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "your_auth_token_here")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+1234567890")

def send_emergency_sms(target_phone_number: str, sos_payload: dict) -> bool:
    """
    Connects to the Twilio API to broadcast the emergency text.
    """
    if TWILIO_ACCOUNT_SID == "your_account_sid_here":
        print("⚠️ TWILIO OFFLINE: Please update src/sms_alert.py with real Twilio credentials to send live texts.")
        return False
        
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # We format a concise version of the SOS for a mobile text message
        sms_body = (
            f"🚨 URGENT: {sos_payload.get('disaster_type', 'HAZARD').upper()} DETECTED 🚨\n"
            f"Severity: {sos_payload.get('severity', 'HIGH')}\n"
            f"Impact Area: {sos_payload['impact_metrics']['area_km2']} km²\n"
            f"Est. People at Risk: {sos_payload['impact_metrics']['est_people']:,}\n"
            f"Action: Check C3 Dashboard immediately."
        )

        message = client.messages.create(
            body=sms_body,
            from_=TWILIO_PHONE_NUMBER,
            to=target_phone_number
        )
        
        print(f"✅ SMS Broadcast Successful! Message ID: {message.sid}")
        return True
        
    except Exception as e:
        print(f"❌ SMS Transmission Failed: {e}")
        return False