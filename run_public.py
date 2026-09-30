import subprocess
import time
import sys
import os
from pyngrok import ngrok
from dotenv import load_dotenv

load_dotenv()

NGROK_TOKEN = os.getenv("NGROK_TOKEN")

if not NGROK_TOKEN:
    print("ERROR: NGROK_TOKEN not found in .env file")
    print("Add this line to .env: NGROK_TOKEN=your_token_here")
    sys.exit(1)

ngrok.set_auth_token(NGROK_TOKEN)

print("Starting Streamlit...")
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py",
     "--server.port=8501",
     "--server.headless=true"],
)

print("Waiting for Streamlit to initialize...")
time.sleep(8)

print("Creating public tunnel...")
tunnel = ngrok.connect(8501)

print("\n" + "="*60)
print(f"  PUBLIC URL : {tunnel.public_url}")
print(f"  LOCAL URL  : http://localhost:8501")
print(f"  Share PUBLIC URL with judges")
print(f"  Works from any device, any network")
print("="*60 + "\n")

print("Press Ctrl+C to stop everything")
try:
    proc.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    proc.terminate()
    ngrok.kill()