# SPDX-License-Identifier: Apache-2.0  
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  
  
import torch  
from vllm.logger import init_logger  
from vllm_gaudi.platform import HpuPlatform  
  
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum  
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum  
  
logger = init_logger(__name__)  
  
  
class HPUOmniPlatform(OmniPlatform, HpuPlatform):  
    """HPU/Gaudi implementation of OmniPlatform.  
  
    Inherits all HPU-specific implementations from vLLM's HpuPlatform,  
    and adds Omni-specific interfaces from OmniPlatform.  
    """  
    _omni_enum = OmniPlatformEnum.HPU  
    @classmethod  
    def activate(cls):  
        """Activate this platform for both vLLM and vLLM-Omni."""  
        # Set vLLM's current_platform to use HPU  
        current_platform.__class__ = cls
    @classmethod  
    def get_omni_ar_worker_cls(cls) -> str:  
        return "vllm_omni.platforms.hpu.worker.hpu_ar_worker.HPUARWorker"  
  
    @classmethod  
    def get_omni_generation_worker_cls(cls) -> str:  
        return "vllm_omni.platforms.hpu.worker.hpu_generation_worker.HPUGenerationWorker"  
  
    @classmethod  
    def get_diffusion_attn_backend_cls(  
        cls,  
        selected_backend: str | None,  
        head_size: int,  
    ) -> str:  
        if selected_backend is not None:  
            backend_upper = selected_backend.upper()  
            backend = DiffusionAttentionBackendEnum[backend_upper]  
            logger.info("Using diffusion attention backend '%s'", backend_upper)  
            return backend.get_path()  
  
        logger.info("Defaulting to diffusion attention backend SDPA")  
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()  
  
    @classmethod  
    def supports_torch_inductor(cls) -> bool:  
        # HPU supports torch.compile with specific optimizations  
        return True  
  
    @classmethod  
    def get_default_stage_config_path(cls) -> str:  
        return "vllm_omni/platforms/hpu/stage_configs"  
  
    @classmethod  
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:  
        if local_rank is None:  
            return torch.device("hpu")  
        return torch.device("hpu", local_rank)  
  
    @classmethod  
    def get_device_count(cls) -> int:  
        return torch.hpu.device_count()  
  
    @classmethod  
    def get_device_version(cls) -> str | None:  
        # HPU driver version  
        try:  
            import habana_frameworks.torch as ht  
            return ht.hpu.get_device_version()  
        except Exception:  
            return None  
  
    @classmethod  
    def synchronize(cls) -> None:  
        torch.hpu.synchronize()  
  
    @classmethod  
    def get_free_memory(cls, device: torch.device | None = None) -> int:  
        # Return the device memory usage in bytes.
        free_hpu_memory, _ = torch.hpu.mem_get_info()
        return free_hpu_memory
    
    @classmethod
    def get_current_memory_usage(cls, device: torch.device | None = None) -> float:  
        """Return current memory usage in bytes for HPU devices."""  
        # Return the device memory usage in bytes.
        print(f"LIBIN DEBUG get_current_memory_usage")
        free_hpu_memory, total_hpu_memory = torch.hpu.mem_get_info()
        return total_hpu_memory - free_hpu_memory
    
    @classmethod  
    @property  
    def dist_backend(cls) -> str:  
        """Return the distributed backend for HPU devices."""  
        return "hccl"