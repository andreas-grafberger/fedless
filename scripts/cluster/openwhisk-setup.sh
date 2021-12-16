#!/usr/bin/env bash

# Installs and configures openwhisk on a fresh kubernetes installation
# Largely based on https://medium.com/@ansjin/openwhisk-deployment-on-a-kubernetes-cluster-7fd3fc2f3726

# Install helm if necessary
if ! command -v helm &>/dev/null; then
  echo "Helm installation not found, installing it now"
  curl https://baltocdn.com/helm/signing.asc | sudo apt-key add -
  sudo apt-get install apt-transport-https --yes
  echo "deb https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
  sudo apt-get update
  sudo apt-get install helm --yes
fi

# Get openwhisk helm chart
git clone https://github.com/apache/openwhisk-deploy-kube.git
cd openwhisk-deploy-kube || exit

# First label all nodes as invokers. Functions will only run on them
# Alternatively use --selector='!node-role.kubernetes.io/master'
kubectl label nodes --all openwhisk-role=invoker --overwrite

# Create cluster config file for helm
cat >mycluster.yaml <<ENDOFFILE
whisk:
  ingress:
    type: NodePort
    apiHostName: $(hostname -I | cut -d' ' -f1)
    apiHostPort: 31001
  limits:
    actions:
      memory:
        max: "5g"
      time:
        max: "10m"
    activation:
      payload:
        max: "128000000"
  containerPool:
    userMemory: "4096m"
  loadbalancer:
    blackbox-fraction: "100%"

k8s:
  persistence:
    enabled: false
nginx:
  httpsNodePort: 31001
invoker:
  containerFactory:
    impl: "kubernetes"
    kubernetes:
      replicaCount: 2
controller:
  replicaCount: 2
metrics:
  prometheusEnabled: true
metrics:
  userMetricsEnabled: true
ENDOFFILE

# Create helm release
helm upgrade owdev ./helm/openwhisk -n openwhisk --create-namespace --install -f mycluster.yaml

## Create new user and delete default guest user
#kubectl -n openwhisk -ti exec owdev-wskadmin -- wskadmin user create admin-1
#if [ "$?" -eq "0" ]; then
#  kubectl -n openwhisk -ti exec owdev-wskadmin -- wskadmin user delete guest
#else
#  echo "Could not create new user..."
#fi
