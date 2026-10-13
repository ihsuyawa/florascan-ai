import time
from pyngrok import ngrok

PORT = 8000

print(f"Connecting ngrok tunnel to local port {PORT}...")

# Open an HTTP tunnel to port 8000 cleanly
tunnel = ngrok.connect(addr=PORT)  # type: ignore
public_url: str = getattr(tunnel, "public_url", "") or ""

print("\n" + "=" * 50)
print(f"   Public Shareable URL: {public_url}")
print("=" * 50 + "\n")
print("Tunnel is live! Press Ctrl+C in this terminal to close.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down tunnel...")
    if public_url:
        ngrok.disconnect(public_url)
    ngrok.kill()
    print("Tunnel closed.")