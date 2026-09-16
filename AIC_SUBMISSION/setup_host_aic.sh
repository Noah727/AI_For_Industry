#!/usr/bin/env bash
set -euo pipefail

# Host setup for the AIC toolkit on Ubuntu 24.04.
# Run this from a normal user account, not as root:
#
#   cd /home/noah/aic-main
#   bash AIC_SUBMISSION/setup_host_aic.sh
#
# The script uses sudo for system packages and installs Pixi into ~/.pixi.

TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
SUDO="sudo"
if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot read /etc/os-release. This setup script expects Ubuntu 24.04."
  exit 1
fi

# shellcheck disable=SC1091
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "Warning: expected Ubuntu 24.04, found ${PRETTY_NAME:-unknown OS}."
fi

echo "==> Installing base apt dependencies"
$SUDO apt-get update
$SUDO apt-get install -y ca-certificates curl gnupg lsb-release

echo "==> Removing conflicting distro Docker packages, if present"
$SUDO apt-get remove -y docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc || true

echo "==> Adding Docker Engine apt repository"
$SUDO install -m 0755 -d /etc/apt/keyrings
$SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
$SUDO chmod a+r /etc/apt/keyrings/docker.asc

ARCH="$(dpkg --print-architecture)"
CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-noble}}"
$SUDO tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${CODENAME}
Components: stable
Architectures: ${ARCH}
Signed-By: /etc/apt/keyrings/docker.asc
EOF

echo "==> Installing Docker Engine and Compose plugin"
$SUDO apt-get update
$SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> Enabling Docker for user ${TARGET_USER}"
$SUDO groupadd docker 2>/dev/null || true
$SUDO usermod -aG docker "$TARGET_USER"
$SUDO systemctl enable docker.service
$SUDO systemctl enable containerd.service
$SUDO systemctl start docker.service

echo "==> Installing Distrobox"
$SUDO apt-get install -y distrobox

echo "==> Adding NVIDIA Container Toolkit apt repository"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

echo "==> Installing NVIDIA Container Toolkit"
$SUDO apt-get update
$SUDO apt-get install -y nvidia-container-toolkit

echo "==> Configuring Docker to use NVIDIA Container Runtime"
$SUDO nvidia-ctk runtime configure --runtime=docker
$SUDO systemctl restart docker

echo "==> Installing Pixi for ${TARGET_USER}"
if [[ ! -x "${TARGET_HOME}/.pixi/bin/pixi" ]]; then
  sudo -u "$TARGET_USER" env HOME="$TARGET_HOME" bash -lc \
    'curl -fsSL https://pixi.sh/install.sh | sh'
fi

echo "==> Pinning Pixi to 0.67.2 for this challenge"
sudo -u "$TARGET_USER" env HOME="$TARGET_HOME" PATH="${TARGET_HOME}/.pixi/bin:${PATH}" \
  pixi self-update --version 0.67.2

echo "==> Installing this repo's Pixi environment"
if [[ -f pixi.toml ]]; then
  sudo -u "$TARGET_USER" env HOME="$TARGET_HOME" PATH="${TARGET_HOME}/.pixi/bin:${PATH}" \
    pixi install
else
  echo "No pixi.toml found in current directory; skipping pixi install."
fi

echo "==> Versions"
docker --version || true
docker compose version || true
distrobox --version || true
nvidia-ctk --version || true
sudo -u "$TARGET_USER" env HOME="$TARGET_HOME" PATH="${TARGET_HOME}/.pixi/bin:${PATH}" \
  pixi --version || true

echo "==> NVIDIA driver check"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo
    echo "nvidia-smi is installed but cannot talk to the NVIDIA driver."
    echo "The container toolkit is installed, but GPU containers will not work until the host NVIDIA driver is fixed."
  fi
else
  echo "nvidia-smi not found. Install/fix the NVIDIA driver before using --nvidia / --gpus all."
fi

cat <<EOF

Setup finished.

Important: log out and log back in, or reboot, so Docker group membership applies.
After that, verify:

  docker run hello-world
  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
  export DBX_CONTAINER_MANAGER=docker
  distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval

EOF
