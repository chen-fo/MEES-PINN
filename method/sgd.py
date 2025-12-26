import jax
import jax.interpreters.xla as _xla
_xla.DeviceArray = jax.Array
import jaxlib.xla_extension as _ext
_all_devices = jax.devices()                
_default_dev = type(_all_devices[0])        

_ext.CpuDevice = _default_dev
_ext.GpuDevice = _default_dev
_ext.TpuDevice = _default_dev
import optax
from jax import numpy as jnp
import sys
sys.path.insert(0, '/home/chenfanke/TaskPINN')

from src.pde.GrayScottEquation import PINN
from src.utils import stack_outputs
from src.nn import BaseNN


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 1





    


