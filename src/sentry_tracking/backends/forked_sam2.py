from __future__ import annotations

from .forked_sam2_runtime import ForkedSAM2Runtime
from .sam2 import SAM2Backend


class ForkedSAM2Backend(SAM2Backend):
    fork = None

    def __init__(self, runtime=None, **runtime_kwargs):
        if runtime is None:
            if self.fork is None:
                raise TypeError("ForkedSAM2Backend subclasses must define 'fork'")
            runtime = ForkedSAM2Runtime(fork=self.fork, **runtime_kwargs)
        super().__init__(runtime=runtime, adapter_name=self.fork)


class SAMURAIBackend(ForkedSAM2Backend):
    fork = "samurai"


class DAM4SAMBackend(ForkedSAM2Backend):
    fork = "dam4sam"


class SAMITEBackend(ForkedSAM2Backend):
    fork = "samite"


class HiM2SAMBackend(ForkedSAM2Backend):
    fork = "him2sam"
