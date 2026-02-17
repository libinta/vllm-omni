# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.workspace import init_workspace_manager
from vllm_gaudi.v1.worker.hpu_worker import HPUWorker

from vllm_omni.platforms.hpu.worker.hpu_generation_model_runner import HPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


class HPUGenerationWorker(OmniWorkerMixin, HPUWorker):
    """NPU generation worker for code2wav stage in Omni model."""

    def init_device(self):
        super().init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)

        self.model_runner = HPUGenerationModelRunner(self.vllm_config, self.device)