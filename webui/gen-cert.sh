#!/usr/bin/env bash
# Generate a self-signed TLS cert for the web UI so browsers allow microphone
# access over the LAN. Re-run this if the Pi's IP changes.
set -e
cd "$(dirname "$0")"

IP=$(hostname -I | awk '{print $1}')
echo "Generating cert for IP ${IP} (+ 127.0.0.1, localhost)..."

mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem -out certs/cert.pem -days 3650 \
  -subj "/CN=voicechat.local" \
  -addext "subjectAltName=IP:${IP},IP:127.0.0.1,DNS:localhost"

echo "Done. Restart the app (./stop.sh && ./start.sh) to pick up the new cert."
