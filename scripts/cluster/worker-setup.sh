#!/usr/bin/env bash

# Script for ubuntu (tested with 18.04) to set up kubernetes and join a network

# Install docker
sudo apt-get update
sudo apt-get install -y docker.io

# Install k8 tools
sudo apt-get install -y apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

echo "Setup complete. Now join the cluster by running the join command returned during cluster setup"
