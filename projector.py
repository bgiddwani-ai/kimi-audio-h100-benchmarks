"""Build and execute only the measured FP16-enabled TensorRT projector."""
import hashlib
import json
from pathlib import Path

import torch
import tensorrt as trt
from safetensors import safe_open

LOGGER=trt.Logger(trt.Logger.WARNING)
ENGINE=Path('/work/engines/projector_fp16_o3.engine')


class Projector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(torch.nn.Linear(5120,3584),torch.nn.GELU(),
            torch.nn.Identity(),torch.nn.Linear(3584,3584),torch.nn.LayerNorm(3584))
    def forward(self,x):return self.layers(x)


class TRTRunner:
    def __init__(self,path):
        self.runtime=trt.Runtime(LOGGER)
        self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:raise RuntimeError('TensorRT deserialization failed')
        self.context=self.engine.create_execution_context()
        self.stream=torch.cuda.Stream()
    def __call__(self,x):
        current=torch.cuda.current_stream();self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            x=x.float().contiguous()
            if not self.context.set_input_shape('audio_features',tuple(x.shape)):
                raise RuntimeError(f'Projector shape outside build profile: {tuple(x.shape)}')
            y=torch.empty(tuple(self.context.get_tensor_shape('embeddings')),device=x.device,dtype=torch.float32)
            self.context.set_tensor_address('audio_features',x.data_ptr())
            self.context.set_tensor_address('embeddings',y.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):raise RuntimeError('TensorRT enqueue failed')
            x.record_stream(self.stream)
        current.wait_stream(self.stream);y.record_stream(current)
        return y


def build():
    fingerprint={'model_revision':'9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b','tensorrt':trt.__version__,
        'gpu_name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
        'builder_level':3,'enabled_precision':'fp16','input_shape':[[1,1,5120],[1,376,5120],[32,1504,5120]]}
    metadata=ENGINE.with_suffix('.json')
    if ENGINE.exists() and metadata.exists():
        old=json.loads(metadata.read_text())
        if old.get('fingerprint')==fingerprint and old.get('sha256')==hashlib.sha256(ENGINE.read_bytes()).hexdigest():
            print('Reusing TensorRT projector',flush=True);return
    ENGINE.parent.mkdir(parents=True,exist_ok=True)
    folder=Path('/work/models/Kimi-Audio-7B-Instruct')
    index=json.loads((folder/'model.safetensors.index.json').read_text())['weight_map']
    weights={}
    for key,file in index.items():
        if key.startswith('model.vq_adaptor.'):
            with safe_open(folder/file,framework='pt',device='cpu') as f:
                weights[key.replace('model.vq_adaptor.','')]=f.get_tensor(key).float()
    model=Projector();model.load_state_dict(weights,strict=True);model.eval()
    onnx=ENGINE.with_suffix('.onnx')
    torch.onnx.export(model,(torch.zeros(1,376,5120),),str(onnx),
        input_names=['audio_features'],output_names=['embeddings'],
        dynamic_axes={'audio_features':{0:'batch',1:'frames'},'embeddings':{0:'batch',1:'frames'}},
        opset_version=18,dynamo=False,external_data=False)
    builder=trt.Builder(LOGGER);network=builder.create_network(0);parser=trt.OnnxParser(network,LOGGER)
    if not parser.parse(onnx.read_bytes()):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config();config.builder_optimization_level=3
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,2<<30);config.set_flag(trt.BuilderFlag.FP16)
    shape=builder.create_optimization_profile()
    shape.set_shape('audio_features',(1,1,5120),(1,376,5120),(32,1504,5120));config.add_optimization_profile(shape)
    serialized=builder.build_serialized_network(network,config)
    if serialized is None:raise RuntimeError('TensorRT build failed')
    temporary=ENGINE.with_suffix('.tmp');temporary.write_bytes(bytes(serialized));temporary.replace(ENGINE)
    runner=TRTRunner(ENGINE)
    (ENGINE.parent/'projector_layers.json').write_text(runner.engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    metadata.write_text(json.dumps({'fingerprint':fingerprint,'sha256':hashlib.sha256(ENGINE.read_bytes()).hexdigest()},indent=2))
    print('Built TensorRT projector',metadata.read_text(),flush=True)


if __name__=='__main__':build()
