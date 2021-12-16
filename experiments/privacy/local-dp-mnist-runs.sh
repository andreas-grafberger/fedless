#!/usr/bin/env bash

# shellcheck disable=SC2043

local_epochs=5
local_batchsize=16
clients_per_round=25
num_dp_rounds=150
num_normal_rounds=100

# Vary l2 norm clip
for l2_norm_clip in 0.5 1.0 5.0 10.0; do
  for noise_multiplier in 1.0; do
    python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs "$num_dp_rounds" --local-epochs "$local_epochs" \
      --local-batch-size "$local_batchsize" --clients-per-round "$clients_per_round" \
      --local-dp --l2-norm-clip "$l2_norm_clip" --noise-multiplier "$noise_multiplier"
  done
done

# Vary noise multiplier
for l2_norm_clip in 1.0; do
  for noise_multiplier in 0.5 1.0 2.0; do
    python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs "$num_dp_rounds" --local-epochs "$local_epochs" \
      --local-batch-size "$local_batchsize" --clients-per-round "$clients_per_round" \
      --local-dp --l2-norm-clip "$l2_norm_clip" --noise-multiplier "$noise_multiplier"
  done
done

# Vary num_microbatches
for num_microbatches in 1 8; do
  python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs "$num_dp_rounds" --local-epochs "$local_epochs" \
    --local-batch-size "$local_batchsize" --clients-per-round "$clients_per_round" \
    --local-dp --l2-norm-clip 1.0 --noise-multiplier 1.0 --num-microbatches "$num_microbatches"
done

# MIA Simulation
python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs "$num_dp_rounds" --local-epochs "$local_epochs" \
  --local-batch-size "$local_batchsize" --clients-per-round "$clients_per_round" \
  --local-dp --l2-norm-clip 1.0 --noise-multiplier 1.0 --num-microbatches 8 --mia

python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs "$num_dp_rounds" --local-epochs "$local_epochs" \
  --local-batch-size "$local_batchsize" --clients-per-round "$clients_per_round" \
  --no-local-dp --mia

# Normal run
python3 -m fedless.benchmark.simulation.fedavg --devices 100 --epochs "$num_normal_rounds" --local-epochs "$local_epochs" \
  --local-batch-size "$local_batchsize" --clients-per-round "$clients_per_round" \
  --no-local-dp
