"""TensorRT replaces only the acoustic projector; vLLM retains the encoder/decoder."""
import os
from projector import TRTRunner
from vllm.model_executor.models.kimi_audio import KimiAudioMultiModalProjector


def forward(self,audio_features):
    if not hasattr(self,'_trt_projector'):
        self._trt_projector=TRTRunner(os.environ['KIMI_TRT_PROJECTOR'])
    return self._trt_projector(audio_features).to(audio_features.dtype)


if os.environ.get('KIMI_TRT_PROJECTOR'):
    KimiAudioMultiModalProjector.forward=forward


class TRTWorkerExtension:
    pass
