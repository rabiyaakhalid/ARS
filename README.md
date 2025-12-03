# Augmented Random Search (ARS)

ARS is a random search method for training linear policies for continuous control problems, based on the paper ["Simple random search provides a competitive approach to reinforcement learning."](https://arxiv.org/abs/1803.07055)

This repo contains an updated and working version of the original Augmented Random Search implementation, adapted to run on a modern Python and Gym setup.

## Install dependencies

We install older NumPy (less than 2) because Gym breaks with NumPy 2.
We also install Ray, Gym, MuJoCo environments and Pandas for reading logs.

```
pip3 install "numpy<2.0" ray gym mujoco "gymnasium[mujoco]" pandas
```

## What was updated in the code

- `ray start` was removed because Ray now starts automatically using `ray.init()`
- `env.seed()` was replaced with `env.reset(seed=...)` because Gym removed the seed method
- `env.spec.timestep_limit` was replaced with `env.spec.max_episode_steps`
- Gym v1 environments like HalfCheetah-v1 are deprecated so we use HalfCheetah-v4
- Fixed compatibility for reset and step return format as Gym now returns terminated and truncated
- Minor fixes for saving policy weights

## Running ARS on Different Environments

We recommend using single threaded linear algebra computations by setting:

```
export MKL_NUM_THREADS=1
```

All arguments passed into ARS are optional and can be modified to train other environments, use different hyperparameters, or use different random seeds.

You can train on multiple MuJoCo tasks. Just change the environment name in the command.

Each environment has a recommended shift value based on the paper.
For each enviroment, we will use the same hyerparamters as per paper.

**Note: Gym version v1 environments have been deprecated and removed from newer Gym and MuJoCo versions hence we will be uisng v4 instead**

### ARS V2

#### Swimmer

```
python3 code/ars.py \
  --env_name Swimmer-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 1 \
  --deltas_used 1 \
  --step_size 0.02 \
  --delta_std 0.01 \
  --shift 0 \
  --filter MeanStdFilter
```

#### Hopper

```
python3 code/ars.py \
  --env_name Hopper-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 4 \
  --deltas_used 4 \
  --step_size 0.02 \
  --delta_std 0.02 \
  --shift 1 \
  --filter MeanStdFilter
```

#### HalfCheetah

```
python3 code/ars.py \
  --env_name HalfCheetah-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 8 \
  --deltas_used 8 \
  --step_size 0.02 \
  --delta_std 0.03 \
  --shift 0 \
  --filter MeanStdFilter
```

#### Walker2d

```
python3 code/ars.py \
  --env_name Walker2d-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 60 \
  --deltas_used 60 \
  --step_size 0.025 \
  --delta_std 0.01 \
  --shift 1 \
  --filter MeanStdFilter
```

#### Ant

```
python3 code/ars.py \
  --env_name Ant-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 40 \
  --deltas_used 40 \
  --step_size 0.01 \
  --delta_std 0.025 \
  --shift 1 \
  --filter MeanStdFilter
```

#### Humanoid

```
python3 code/ars.py \
  --env_name Humanoid-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 230 \
  --deltas_used 230 \
  --step_size 0.02 \
  --delta_std 0.0075 \
  --shift 5 \
  --filter MeanStdFilter
```

### ARS V2 t

#### Swimmer

```
python3 code/ars.py \
  --env_name Swimmer-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 1 \
  --deltas_used 1 \
  --step_size 0.02 \
  --delta_std 0.01 \
  --shift 0 \
  --filter MeanStdFilter
```

#### Hopper

```
python3 code/ars.py \
  --env_name Hopper-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 8 \
  --deltas_used 4 \
  --step_size 0.01 \
  --delta_std 0.025 \
  --shift 1 \
  --filter MeanStdFilter
```

#### HalfCheetah

```
python3 code/ars.py \
  --env_name HalfCheetah-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 32 \
  --deltas_used 4 \
  --step_size 0.02 \
  --delta_std 0.03 \
  --shift 0 \
  --filter MeanStdFilter
```

#### Walker2d

```
python3 code/ars.py \
  --env_name Walker2d-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 40 \
  --deltas_used 30 \
  --step_size 0.03 \
  --delta_std 0.025 \
  --shift 1 \
  --filter MeanStdFilter
```

#### Ant

```
python3 code/ars.py \
  --env_name Ant-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 60 \
  --deltas_used 20 \
  --step_size 0.015 \
  --delta_std 0.025 \
  --shift 1 \
  --filter MeanStdFilter
```

#### Humanoid

```
python3 code/ars.py \
  --env_name Humanoid-v4 \
  --n_iter 1000 \
  --n_workers 16 \
  --n_directions 230 \
  --deltas_used 230 \
  --step_size 0.02 \
  --delta_std 0.0075 \
  --shift 5 \
  --filter MeanStdFilter
```

For Humanoid the V2 t entries in the table have N equal to b so V2 and V2 t coincide.

## ARS Directional (Evolution-Path Biased ARS)

`code/ars_directional.py` extends the vanilla algorithm with an evolution-path memory. Instead of sampling completely new perturbations each iteration, the learner keeps an exponential moving average of recent updates (the “evolution path”) and blends future noise directions toward that path. This mild directional bias reuses the trends of past successful updates, smoothing convergence without switching to a full optimizer such as Adam.

Key extras compared to `code/ars.py`:

- `--directional_alpha`: how much of each perturbation should align with the evolution path (0.0 means pure isotropic noise).
- `--directional_beta`: decay factor for the path; smaller values forget past updates faster.
- `--directional_min_norm`: minimal path norm before bias activates (prevents early noise from causing bias).
- `--directional_warmup_iters`: number of iterations to run with pure isotropic exploration before turning on directional bias. Set this to a few hundred steps if you want early exploration to match the original ARS behavior.

Whenever the bias is active, the worker and learner deterministically transform the shared-noise deltas so mirrored rollouts remain reproducible. Evaluation rollouts always run without directional noise.

### Example: Hopper with Directional Bias

```
python3 code/ars_directional.py \
  --env_name Hopper-v4 \
  --n_iter 2000 \
  --n_workers 32 \
  --n_directions 320 \
  --deltas_used 160 \
  --step_size 0.02 \
  --delta_std 0.02 \
  --shift 1 \
  --filter MeanStdFilter \
  --directional_alpha 0.2 \
  --directional_beta 0.8 \
  --directional_min_norm 1e-5 \
  --directional_warmup_iters 200 \
  --dir_path results/directional/Hopper
```

### Example: Walker2d with Warm-up

```
python3 code/ars_directional.py \
  --env_name Walker2d-v4 \
  --n_iter 2000 \
  --n_workers 32 \
  --n_directions 320 \
  --deltas_used 160 \
  --step_size 0.02 \
  --delta_std 0.03 \
  --shift 1 \
  --filter MeanStdFilter \
  --directional_alpha 0.15 \
  --directional_beta 0.6 \
  --directional_min_norm 1e-5 \
  --directional_warmup_iters 400 \
  --dir_path results/directional/Walker
```

### Additional Directional Recipes

**HalfCheetah**

```
python3 code/ars_directional.py \
  --env_name HalfCheetah-v4 \
  --n_iter 2000 \
  --n_workers 32 \
  --n_directions 320 \
  --deltas_used 160 \
  --step_size 0.02 \
  --delta_std 0.03 \
  --shift 0 \
  --filter MeanStdFilter \
  --directional_alpha 0.15 \
  --directional_beta 0.7 \
  --directional_min_norm 1e-5 \
  --directional_warmup_iters 300 \
  --dir_path results/directional/HalfCheetah
```

**Ant**

```
python3 code/ars_directional.py \
  --env_name Ant-v4 \
  --n_iter 2000 \
  --n_workers 32 \
  --n_directions 320 \
  --deltas_used 160 \
  --step_size 0.015 \
  --delta_std 0.025 \
  --shift 1 \
  --filter MeanStdFilter \
  --directional_alpha 0.15 \
  --directional_beta 0.7 \
  --directional_min_norm 1e-5 \
  --directional_warmup_iters 300 \
  --dir_path results/directional/Ant
```

**Humanoid**

```
python3 code/ars_directional.py \
  --env_name Humanoid-v4 \
  --n_iter 3000 \
  --n_workers 64 \
  --n_directions 400 \
  --deltas_used 200 \
  --step_size 0.02 \
  --delta_std 0.0075 \
  --shift 5 \
  --filter MeanStdFilter \
  --directional_alpha 0.1 \
  --directional_beta 0.8 \
  --directional_min_norm 1e-5 \
  --directional_warmup_iters 500 \
  --dir_path results/directional/Humanoid
```

Feel free to tweak `directional_alpha`, warm-up length, and `delta_std` per task—the recipes above mirror the paper’s base hyperparameters while adding directional bias once the warm-up window passes.
