#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Switching to project root directory $ROOT_DIR"
cd "$ROOT_DIR"

# Build python project
echo "Build package"
python setup.py bdist_wheel

# Building image
echo "Build and Push Image"
docker build -f ./images/openwhisk/base/Dockerfile -t fedless-openwhisk  .
docker tag fedless-openwhisk andreasgrafberger/fedless-openwhisk:clients
docker push andreasgrafberger/fedless-openwhisk:clients

