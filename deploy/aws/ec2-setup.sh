#!/bin/bash
# Run once on a fresh Ubuntu 24.04 EC2 instance:  bash ec2-setup.sh
set -e
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl unzip
# 4 GB swap: the 8 GB instance needs headroom while building images and loading two models
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
fi
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
echo "Docker installed. Log out and back in (or run: newgrp docker), then:"
echo "  cd ~/signature-platform && cp deploy/aws/aws.env.example deploy/aws/aws.env && nano deploy/aws/aws.env"
echo "  docker compose -f deploy/aws/docker-compose.aws.yml --env-file deploy/aws/aws.env up -d --build"
