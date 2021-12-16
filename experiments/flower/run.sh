#!/usr/bin/env bash

set -e

server_ssh_host="<TODO>"
server_ip="<TODO>"
port="31532"
server_address="$server_ip:$port"
server_cpus="16"
server_memory="32g"
rounds=100
min_num_clients=100
num_clients_total=100

client_cpus=2.0
client_memory="8g"
dataset="mnist"
batch_size=10
epochs=5
optimizer="Adam"
lr=0.001

session_id="$RANDOM"

# Syntax='ssh-host;clients'
workers[0]='<TODO>;30'
workers[1]='<TODO>;15'
workers[2]='<TODO>;15'
workers[3]='<TODO>;5'
workers[4]='<TODO>;5'
workers[5]='<TODO>;5'
workers[6]='<TODO>;5'
workers[7]='<TODO>;20'

echo "Update server and client images (build and push)"
docker build -f server.Dockerfile -t "flower-server" .
docker tag "flower-server" andreasgrafberger/flower:server
docker push andreasgrafberger/flower:server
docker build -f client.Dockerfile -t "flower-client" .
docker tag "flower-client" andreasgrafberger/flower:client
docker push andreasgrafberger/flower:client

echo "Making sure all client machines have the latest docker images and no running clients"
ssh "$server_ssh_host" "docker pull andreasgrafberger/flower:server" >/dev/null
for worker in "${workers[@]}"; do
  # shellcheck disable=SC2206
  worker_info=(${worker//;/ })
  ssh_host=${worker_info[0]}
  echo "Checking $ssh_host"
  ssh "$ssh_host" "docker pull andreasgrafberger/flower:client" >/dev/null
  ssh "$ssh_host" "docker ps | grep andreasgrafberger/flower:client | cut -d ' ' -f 1 | xargs -r docker stop" >/dev/null
  (ssh "$ssh_host" "sudo usermod -aG docker \$USER " || true) &>/dev/null
  ssh "$ssh_host" "mkdir -p flower-logs" >/dev/null
done

run_experiment() {
  dataset=$1
  min_num_clients=$2
  client_cpus=$3
  client_memory=$4
  batch_size=$5
  epochs=$6
  optimizer=$7
  lr=$8
  rounds=$9
  session_id=${10}

  echo "Experiment: dataset=$dataset, min_num_clients=$min_num_clients, client_cpus=$client_cpus,
 client_memory=$client_memory, dataset=$dataset, batch_size=$batch_size, epochs=$epochs,
 rounds=$rounds, optimizer=$optimizer, lr=$lr"

  echo "Removing running container if it exists..."
  ssh "$server_ssh_host" 'docker stop fl-server' ||
    true &>/dev/null

  exp_filename="flower-logs/fedless_${dataset}_${min_num_clients}_${num_clients_total}_${epochs}_${session_id}"

  echo "Starting server, results are stored in $exp_filename.out and $exp_filename.err"
  run_cmd="docker run --rm -p $port:$port --name fl-server \
-e https_proxy=\$http_proxy \
--cpus $server_cpus --memory $server_memory --memory-swap $server_memory \
andreasgrafberger/flower:server --rounds $rounds --min-num-clients $min_num_clients --dataset=$dataset"
  ssh "$server_ssh_host" "mkdir -p flower-logs" >/dev/null
  # shellcheck disable=SC2029
  ssh "$server_ssh_host" "nohup $run_cmd > $exp_filename.out 2> $exp_filename.err < /dev/null &" >/dev/null

  echo "Deploying and starting clients..."

  current_partition=0

  echo "Starting clients..."
  for worker in "${workers[@]}"; do
    # shellcheck disable=SC2206
    worker_info=(${worker//;/ })
    ssh_host=${worker_info[0]}
    cores_assigned_to_host=${worker_info[1]}
    if [[ $current_partition -ge $num_clients_total ]]; then
      break
    fi
    echo "Starting $cores_assigned_to_host clients on $ssh_host"
    for ((i = 1; i <= cores_assigned_to_host; i++)); do
      if [[ $current_partition -ge $num_clients_total ]]; then
        break
      fi
      run_cmd="docker run --rm \
--network host \
--cpus $client_cpus \
--memory $client_memory \
--memory-swap $client_memory \
-e https_proxy=\$http_proxy \
-e no_proxy=$server_ip \
andreasgrafberger/flower:client \
--server $server_address \
--dataset $dataset \
--partition $current_partition \
--batch-size $batch_size \
--epochs $epochs \
--optimizer $optimizer \
--lr $lr \
--clients-total $num_clients_total"
      echo "($ssh_host) $run_cmd"
      # shellcheck disable=SC2029
      ssh "$ssh_host" "nohup $run_cmd > ${exp_filename}_$current_partition.out 2> ${exp_filename}_$current_partition.err < /dev/null &"
      current_partition=$((current_partition + 1))
    done
  done

  if ((current_partition >= num_clients_total)); then
    echo "Successfully deployed all clients"
  else
    echo "WARNING: Tried to deploy client partition ($current_partition / $num_clients_total) but no compute left..."
  fi
  sleep 10
  while ssh "$server_ssh_host" "docker ps | grep andreasgrafberger/flower:server"; do
    echo "Not finished yet"
    sleep 30
  done
  sleep 10
  wait
}

#run_experiment dataset min_num_clients client_cpus client_memory batch_size epochs optimizer lr rounds session_id

## MNIST
run_experiment "mnist" 75 2.0 "8g" 10 5 "Adam" 0.001 100 "$RANDOM"
run_experiment "mnist" 75 2.0 "8g" 10 5 "Adam" 0.001 100 "$RANDOM"
run_experiment "mnist" 75 2.0 "8g" 10 5 "Adam" 0.001 100 "$RANDOM"
#
## Femnist
run_experiment "femnist" 75 2.0 "8g" 10 5 "Adam" 0.001 100 "$RANDOM"
run_experiment "femnist" 75 2.0 "8g" 10 5 "Adam" 0.001 100 "$RANDOM"
run_experiment "femnist" 75 2.0 "8g" 10 5 "Adam" 0.001 100 "$RANDOM"

# Shakespeare
run_experiment "shakespeare" 25 4.0 "8g" 32 1 "SGD" 0.8 100 "$RANDOM"
run_experiment "shakespeare" 25 4.0 "8g" 32 1 "SGD" 0.8 100 "$RANDOM"
run_experiment "shakespeare" 25 4.0 "8g" 32 1 "SGD" 0.8 100 "$RANDOM"
