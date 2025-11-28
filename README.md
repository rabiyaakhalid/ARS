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

You can train on multiple MuJoCo tasks. Just change the environment name in the command.

Each environment has a recommended shift value based on the paper. Some of the enviroments are:

### HalfCheetah

```
python3 code/ars.py --env_name HalfCheetah-v4 --shift 0 --n_iter 50 --n_workers 4
```
### Swimmer

```
python3 code/ars.py --env_name Swimmer-v4 --shift 0 --n_iter 50 --n_workers 4
```

### Hopper

```
python3 code/ars.py --env_name Swimmer-v4 --shift 0 --n_iter 50 --n_workers 4
```

To run longer training that matches the paper more closely, increase the number of iterations to around one thousand.

#### Example

```
python3 code/ars.py --env_name Hopper-v4 --shift 1 --n_iter 50 --n_workers 4
```
**You can replace HalfCheetah-v4 with any of the other environments.**

