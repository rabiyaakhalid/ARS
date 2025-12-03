# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# OPTIMIZERS FOR MINIMIZING OBJECTIVES
class Optimizer(object):
    def __init__(self, w_policy):
        self.w_policy = w_policy.flatten()
        self.dim = w_policy.size
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        ratio = np.linalg.norm(step) / (np.linalg.norm(self.w_policy) + 1e-5)
        return self.w_policy + step, ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def _compute_step(self, globalg):
        step = -self.stepsize * globalg
        return step


class Adam(Optimizer):
    """
    Adam optimizer for ARS V3.
    Implements adaptive learning rates with momentum.
    """
    def __init__(self, pi, stepsize, beta1=0.9, beta2=0.999, eps=1e-8):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(self.dim, dtype=np.float64)
        self.v = np.zeros(self.dim, dtype=np.float64)

    def _compute_step(self, globalg):
        # Increment t first (since we're called directly, not through update())
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg ** 2)
        
        # Bias correction: avoid division by zero when t=0 (shouldn't happen now, but safety check)
        if self.t == 0:
            m_hat = self.m
            v_hat = self.v
        else:
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
        
        step = -self.stepsize * m_hat / (np.sqrt(v_hat) + self.eps)
        
        # Check for NaN or Inf values
        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            print(f"Warning: NaN/Inf detected in Adam step, using SGD fallback")
            step = -self.stepsize * globalg
        
        return step


class AdamTuned(Optimizer):
    """
    Adam optimizer tuned specifically for ARS.
    
    Key differences from standard Adam:
    - Lower beta1 (0.5) for less momentum (ARS gradients are noisy estimates)
    - Higher epsilon (1e-4) for more stability
    - Optional gradient scaling to match SGD step magnitude
    """
    def __init__(self, pi, stepsize, beta1=0.5, beta2=0.99, eps=1e-4, scale_to_sgd=True):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.scale_to_sgd = scale_to_sgd
        self.m = np.zeros(self.dim, dtype=np.float64)
        self.v = np.zeros(self.dim, dtype=np.float64)

    def _compute_step(self, globalg):
        self.t += 1
        
        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Compute Adam step
        adam_step = m_hat / (np.sqrt(v_hat) + self.eps)
        
        if self.scale_to_sgd:
            # Scale Adam step to have similar magnitude to SGD step
            # This preserves Adam's direction but uses SGD's scale
            sgd_norm = np.linalg.norm(globalg)
            adam_norm = np.linalg.norm(adam_step)
            if adam_norm > 1e-8:
                scale = sgd_norm / adam_norm
                adam_step = adam_step * scale
        
        step = -self.stepsize * adam_step
        
        # Safety check
        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            print(f"Warning: NaN/Inf detected, using SGD fallback")
            step = -self.stepsize * globalg
        
        return step


class SGDWithMomentum(Optimizer):
    """
    SGD with momentum for smoother updates.
    """
    def __init__(self, pi, stepsize, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.momentum = momentum
        self.velocity = np.zeros(self.dim, dtype=np.float64)

    def _compute_step(self, globalg):
        self.velocity = self.momentum * self.velocity - self.stepsize * globalg
        return self.velocity

