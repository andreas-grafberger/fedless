#!/usr/bin/env bash

set -e

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for i in {1..100}; do
  function_name="http-${i}"
  echo "Deploying function $function_name"
  # shellcheck disable=SC2140
  gcloud functions deploy "$function_name" \
    --runtime python38 \
    --trigger-http \
    --entry-point="http" \
    --allow-unauthenticated \
    --memory=2048MB \
    --timeout=540s \
    --region europe-west3 \
    --max-instances 50 \
    --set-env-vars TF_ENABLE_ONEDNN_OPTS=1 &
  if [ $((i % 15)) -eq 0 ]; then
    echo "Waiting for previous functions to finish deployment"
    wait
  fi
done
wait
