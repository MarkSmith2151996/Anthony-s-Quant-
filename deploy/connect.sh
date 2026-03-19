#!/usr/bin/env bash
# =============================================================================
# Quick SSH connect script
# Edit the variables below after you create your VM
# Usage: bash deploy/connect.sh
# =============================================================================

# ---- EDIT THESE ----
VM_IP="YOUR_VM_IP_HERE"
VM_USER="ubuntu"
SSH_KEY="~/.ssh/id_rsa"
# --------------------

if [ "$VM_IP" = "YOUR_VM_IP_HERE" ]; then
    echo "ERROR: Edit deploy/connect.sh and set your VM_IP first!"
    echo ""
    echo "After creating your VM, update these values:"
    echo "  VM_IP   = your VM's public IP"
    echo "  VM_USER = ubuntu (Oracle) or your username (GCP)"
    echo "  SSH_KEY = path to your SSH private key"
    exit 1
fi

ssh -i "$SSH_KEY" "${VM_USER}@${VM_IP}"
