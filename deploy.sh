#!/bin/bash

SERVICE=$1     # Only 'retrieval' is supported now
ENV=$2         # test | prod
PORT=$3        # unused (can be inferred from override)
MODULE=$4      # e.g., cpi_top5_results_v5_vm_experimental_citeurl

cd /mnt/data/sujalmh/vr
git pull origin $ENV

# Kill existing container if running
docker compose -f docker-compose.yml -f docker-compose.override.${ENV}.yml down || true

# Build and run only retrieval service
docker compose -f docker-compose.yml -f docker-compose.override.${ENV}.yml up -d --build retrieval