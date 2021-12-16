#!/usr/bin/env bash

set -e

trap "exit" INT TERM ERR
trap "kill 0" EXIT

for i in {1..200}; do
  function_name="http-indep-secure-${i}"
  echo "Deploying function $function_name"
  # shellcheck disable=SC2140
  gcloud functions deploy "$function_name" \
    --runtime python38 \
    --trigger-http \
    --entry-point="http" \
    --allow-unauthenticated \
    --memory=2048MB \
    --timeout=540s \
    --max-instances 50 \
    --region europe-west3 \
    --set-env-vars TF_ENABLE_ONEDNN_OPTS=1,COGNITO_USER_POOL_REGION="$COGNITO_USER_POOL_REGION",COGNITO_USER_POOL_ID="$COGNITO_USER_POOL_ID",COGNITO_INVOKER_CLIENT_ID="$COGNITO_INVOKER_CLIENT_ID",COGNITO_REQUIRED_SCOPE="client-functions/invoke" &
  if [ $((i % 15)) -eq 0 ]; then
    echo "Waiting for previous functions to finish deployment"
    wait
  fi
done
wait
