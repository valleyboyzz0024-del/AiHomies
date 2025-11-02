"""
Elite AI Assistant - Desktop Application
Launches the Flask app in a native desktop window
"""

import webview
import threading
import time
import sys
import os

# Import the main Flask app
from elite_ai_assistant import app, socketio, assistant

def start_flask_server():
    """Start Flask server in background thread"""
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))

    print("=" * 60)
    print(">> Elite AI Desktop App Starting...")
    print("=" * 60)
    print(f"[*] Server: {host}:{port}")
    print(f"[*] Opening native window...")
    print("")
    print("[+] Loaded API Keys:")

    keys_found = False
    for provider, key in assistant.api_keys.items():
        if key:
            masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
            print(f"    [OK] {provider.upper()}: {masked_key}")
            keys_found = True
        else:
            print(f"    [ ] {provider.upper()}: Not set")

    if not keys_found:
        print("")
        print("[!] WARNING: No API keys found in .env file!")

    print("")
    print("=" * 60)
    print("[>] Desktop app is now running!")
    print("=" * 60)
    print("")

    # Start Flask-SocketIO server
    socketio.run(app, debug=False, port=port, host=host, use_reloader=False)

def on_closed():
    """Handle window close event"""
    print("\n[*] Desktop app closed by user")
    print("[*] Shutting down...")

    # Give Flask time to cleanup
    time.sleep(1)
    sys.exit(0)

if __name__ == '__main__':
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Start Flask server in background thread
    server_thread = threading.Thread(target=start_flask_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(2)

    # Create native desktop window
    window = webview.create_window(
        '🚀 Elite AI Assistant',
        'http://127.0.0.1:5000',
        width=1400,
        height=900,
        resizable=True,
        frameless=False,
        easy_drag=True,
        background_color='#000000'
    )

    # Start the desktop app
    webview.start(on_closed)
