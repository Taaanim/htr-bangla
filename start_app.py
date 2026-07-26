"""
start_app.py - Unified Launcher for Bangla HTR Web Application

Launches the Express API Server (Node.js) on port 5001 and
the Vite React Frontend (Web App UI) on port 5173.
"""

import os
import sys
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(PROJECT_ROOT, "web_app", "server")
CLIENT_DIR = os.path.join(PROJECT_ROOT, "web_app", "client")

def main():
    print("=" * 60)
    print("STARTING BANGLA HTR WEB APPLICATION")
    print("=" * 60)

    # 1. Start Express Server
    print("Starting Express API Server (port 5001)...")
    server_proc = subprocess.Popen(["node", "server.js"], cwd=SERVER_DIR)

    time.sleep(2)

    # 2. Start Vite Client Server
    print("Starting React Frontend Dev Server (port 5173)...")
    client_proc = subprocess.Popen(["npx", "vite"], cwd=CLIENT_DIR)

    print("\n" + "=" * 60)
    print("🚀 Web Application is RUNNING!")
    print("  • Frontend UI : http://localhost:5173")
    print("  • API Server  : http://localhost:5001")
    print("=" * 60)
    print("Press Ctrl+C to stop all servers.\n")

    try:
        server_proc.wait()
        client_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping web servers...")
        server_proc.terminate()
        client_proc.terminate()

if __name__ == "__main__":
    main()
