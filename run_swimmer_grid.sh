#!/usr/bin/env bash

ENV="Swimmer-v4"

ALPHAS=(0.01 0.02 0.025)
NOISES=(0.03 0.02 0.01)
SEEDS=(5 10 15 20 55)

N=1
B=1
ITER=1000
WORKERS=16



for SEED in SEEDS
do
  python3.10 code/ars.py \
    --env_name "${ENV}" \
    --n_iter 1000 \
    --n_workers 16 \
    --n_directions 1 \
    --deltas_used 1 \
    --step_size 0.02 \
    --delta_std 0.01 \
    --filter MeanStdFilter \
    --shift 0 \
    --seed ${SEED} \
    > results/V1/Swimmer/swimmer_seed${SEED}.txt
done
