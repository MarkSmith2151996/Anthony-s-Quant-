#!/usr/bin/env bash
# Install BerjQuant as a systemd service so it runs on boot
# Usage: bash deploy/install-service.sh
set -euo pipefail

USER=$(whoami)
SERVICE_FILE="deploy/berjquant.service"
DEST="/etc/systemd/system/berjquant@.service"

echo "Installing BerjQuant service for user: $USER"

# Substitute the template and install
sudo cp "$SERVICE_FILE" "$DEST"
sudo systemctl daemon-reload
sudo systemctl enable "berjquant@${USER}"
sudo systemctl start "berjquant@${USER}"

echo "Service installed and started."
echo ""
echo "Useful commands:"
echo "  sudo systemctl status berjquant@${USER}   # Check status"
echo "  sudo journalctl -u berjquant@${USER} -f   # Follow logs"
echo "  sudo systemctl restart berjquant@${USER}   # Restart"
echo "  sudo systemctl stop berjquant@${USER}      # Stop"
