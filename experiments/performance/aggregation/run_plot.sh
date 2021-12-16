#!/usr/bin/env bash

mprof run fedavg-memory.py --stream --large-models
mprof plot --output "mem-usage-stream-large.pdf" --flame -t "Streaming FedAvg"

mprof run fedavg-memory.py --stream --large-models --chunk-size 1
mprof plot --output "mem-usage-stream-large-1.pdf" --flame -t "Streaming FedAvg (1)"

mprof run fedavg-memory.py --stream --large-models --chunk-size 10
mprof plot --output "mem-usage-stream-large-10.pdf" --flame -t "Streaming FedAvg (10)"

mprof run fedavg-memory.py --stream --large-models --chunk-size 20
mprof plot --output "mem-usage-stream-large-20.pdf" --flame -t "Streaming FedAvg (20)"

mprof run fedavg-memory.py --no-stream --large-models
mprof plot --output "mem-usage-no-stream-large.pdf" --flame -t "Vanilla FedAvg"
