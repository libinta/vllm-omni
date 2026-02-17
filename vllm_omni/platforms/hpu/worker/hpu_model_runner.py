# SPDX-License-Identifier: Apache-2.0  
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  
  
import math  
from typing import TYPE_CHECKING, Any, cast  
  
import numpy as np  
import torch  
from vllm.config import CUDAGraphMode  
from vllm.distributed import get_tensor_model_parallel_world_size  
from vllm.distributed.parallel_state import get_pp_group  
from vllm.logger import init_logger  
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding  
from vllm.model_executor.models.interfaces import supports_mrope  
from vllm.model_executor.models.interfaces_base import VllmModelForPooling  
from vllm.sampling_params import SamplingType  
from vllm.sequence import IntermediateTensors  
from vllm.utils.math_utils import cdiv  
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
  
from vllm_omni.model_executor.models.output_templates import OmniOutput  
  
if TYPE_CHECKING:  
    from vllm.v1.core.sched.output import SchedulerOutput  
  
logger = init_logger(__name__)  
  
  
class OmniHPUModelRunner(HPUModelRunner):  
    """Base HPU model runner for vLLM-Omni."""  
      
    def __init__(self, *args, **kwargs):  # Add *args, **kwargs here  
        super().__init__(*args, **kwargs)  
        self._omni_per_req_additional_information: dict[str, dict] | None = None  
        self._omni_num_scheduled_tokens_np: np.ndarray | None = None  
        self._omni_last_model_output: object | None = None
