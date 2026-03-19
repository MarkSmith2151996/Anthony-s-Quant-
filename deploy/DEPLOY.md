# Deploying BerjQuant to a Free Cloud VM

## Option A: Oracle Cloud (Recommended — Always Free)

### 1. Create an account
- Go to https://www.oracle.com/cloud/free/
- Sign up (credit card required for verification, but won't be charged)

### 2. Create a VM
- Go to **Compute → Instances → Create Instance**
- Image: **Ubuntu 22.04** (or 24.04)
- Shape: **VM.Standard.A1.Flex** (ARM) — 4 OCPUs, 24GB RAM (free forever)
- Add your SSH public key
- Create

### 3. Connect and setup
```bash
# From your local machine:
ssh -i ~/.ssh/your_key ubuntu@YOUR_VM_IP

# On the VM:
git clone https://github.com/bngapple/berjquant.git
cd berjquant
bash deploy/setup-vm.sh
```

### 4. Run persistently
```bash
# Option A: tmux (simple)
tmux new -s quant
source .venv/bin/activate
python run_pipeline.py
# Ctrl+B, D to detach — process keeps running

# Option B: systemd service (auto-restart on reboot)
bash deploy/install-service.sh
```

### 5. Use Claude Code on the VM
```bash
ssh -i ~/.ssh/your_key ubuntu@YOUR_VM_IP
cd ~/berjquant
source .venv/bin/activate
claude   # starts Claude Code — works exactly like local
```

---

## Option B: Google Cloud (Always Free e2-micro)

### 1. Create an account
- Go to https://cloud.google.com/free
- $300 free credits + always-free e2-micro

### 2. Create a VM
```bash
# Using gcloud CLI:
gcloud compute instances create berjquant-vm \
    --machine-type=e2-micro \
    --zone=us-central1-a \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=30GB
```

### 3. Connect and setup
```bash
gcloud compute ssh berjquant-vm
# Then same setup steps as Oracle
```

---

## After Setup: Daily Workflow

```bash
# Connect to your VM
ssh ubuntu@YOUR_VM_IP

# Start Claude Code
cd ~/berjquant && source .venv/bin/activate
claude

# Or attach to running session
tmux attach -t quant
```

## Updating Code

```bash
cd ~/berjquant
git pull
source .venv/bin/activate
pip install -r requirements.txt
# Restart service if using systemd:
sudo systemctl restart berjquant@$(whoami)
```
