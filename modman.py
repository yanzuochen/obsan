from statistics import mode
import torch
import numpy as np
import os
import time
import support.classifier as models
import tvm
from tvm import tir
from collections import namedtuple
import onnx
from scipy.special import softmax
from tqdm import tqdm

from tvm.contrib.debugger import debug_executor
from tvm.contrib import graph_executor

from tvm.contrib import relay_viz
from tvm.contrib.relay_viz.dot import DotPlotter
from tvm.contrib.relay_viz.interface import DefaultVizParser

import utils

data_root = '/export/d1/alex/aesan-data'
targets = {
    'llvm': 'llvm',
    'avx2': 'llvm -mcpu=core-avx2',
    'avx2-cblas': 'llvm -mcpu=core-avx2 -libs=cblas',
}
target = targets['avx2']
dev = tvm.device(str(target), 0)

DEFAULT_OUTPUT_DEFS = [
    {'dtype': 'float32', 'shape': [1, 10]},
]

DEFAULT_COV_OUTPUT_DEFS = DEFAULT_OUTPUT_DEFS + [
    {'dtype': 'float32', 'shape': [1]},
]

IRModPack = namedtuple('IRModPack', ['irmod', 'params'])
ModRunResult = namedtuple('ModRunResult', ['outputs', 'exec_time', 'perf'])

def ensure_output_defs(output_defs):
    return [
        {
            'dtype': d['dtype'],
            'shape': [x.value if isinstance(x, tir.IntImm) else x for x in d['shape']]
        }
        for d in output_defs
    ]

def make_output_bufs(output_defs, zero=False):
    output_defs = ensure_output_defs(output_defs)
    if zero:
        return [tvm.nd.array(np.zeros(**d)) for d in output_defs]
    return [tvm.nd.empty(**d) for d in output_defs]

class WrappedRtMod:
    def __init__(self, rtmod, output_defs, input_name='input0') -> None:
        self.rtmod = rtmod
        self.output_defs = ensure_output_defs(output_defs)
        self.input_name = input_name
        self.output_bufs = [tvm.nd.empty(**d) for d in self.output_defs]

    def run(self, data, rettype='cov'):
        assert rettype in {'cov', 'pred', 'all'}
        rtmod = self.rtmod
        rtmod.set_input(self.input_name, data)
        rtmod.run()
        if rettype == 'cov':
            return [rtmod.get_output(i+1, buf).numpy()
                    for i, buf in enumerate(self.output_bufs[1:])]
        elif rettype == 'pred':
            return rtmod.get_output(0, self.output_bufs[0]).numpy()
        else:
            return [rtmod.get_output(i, buf).numpy()
                    for i, buf in enumerate(self.output_bufs)]

def load_bounds(model_name, dataset, mode, data_path=None) -> list:
    if not data_path:
        data_path = f'{data_root}/Coverage/{dataset}-{model_name}-{mode}.pth'
    data = torch.load(data_path, map_location=torch.device('cpu'))
    if isinstance(data, list):
        return data
    assert 'range' in data
    # We rely on the order of the dict here
    ret = []
    for lows, highs in data['range'].values():
        ret.append(torch.stack([lows, highs]).numpy())
    return ret

def load_covs(model_name, dataset, mode, data_path=None) -> list:
    if not data_path:
        data_path = f'{data_root}/Coverage/{dataset}-{model_name}-{mode}.pth'
    data = torch.load(data_path, map_location=torch.device('cpu'))
    if isinstance(data, list):
        return data
    # We rely on the order of the dict here
    return [v.numpy() for v in data.values()]

def create_rtmod(libmod, debug=False, debug_dump_root=None):
    dev = tvm.device('cpu')
    if not debug:
        return graph_executor.GraphModule(libmod["default"](dev))
    return debug_executor.create(libmod['get_graph_json'](), libmod, dev, dump_root=debug_dump_root)

def get_torch_mod(model_name, dataset):
    model_class = getattr(models, model_name)
    torch_model = model_class(pretrained=False)
    torch_model.eval()
    torch_model.load_state_dict(torch.load(f'{data_root}/models/{dataset}/{model_name}/{model_name}.pt'))
    return torch_model

def export_torch_mod(model, input_shape, fname, optimise=False):
    x = torch.randn(*input_shape, requires_grad=False, dtype=torch.float32)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    opt_args = {}
    if not optimise:
        opt_args['training'] = torch.onnx.TrainingMode.TRAINING
        opt_args['do_constant_folding'] = False
    torch.onnx.export(model, x, fname, **opt_args,
            export_params=True, opset_version=14,
            input_names=['input0'])

def save_irmod_viz(irmod, basename):
    viz = relay_viz.RelayVisualizer(irmod, plotter=DotPlotter(), parser=DefaultVizParser())
    viz.render(basename)

def extra_params_removed(params_dict):
    return {k: v for k, v in params_dict.items() if not k.startswith('__ep_')}

def extra_params_zeroed(params_dict):
    return {k: (v if not k.startswith('__ep_') else tvm.nd.array(np.zeros_like(v)))
            for k, v in params_dict.items()}

def create_zeroed_extra_params_dict(extra_params_vars, ones=False):
    fill_fn = np.ones if ones else np.zeros
    ep_types = [utils.get_type(v) for v in extra_params_vars]
    ep_defs = [{'shape': t.concrete_shape, 'dtype': t.dtype} for t in ep_types]
    return {f'__ep_{i}': tvm.nd.array(fill_fn(**d))
            for i, d in enumerate(ep_defs)}

def torch2irmod(torch_model, input_shape, input_name='input0'):
    # trace instead of script: https://discuss.tvm.apache.org/t/model-with-torch-where-function-breaks-tvm-interpreter/8069/4
    torch_model.eval()
    scripted_model = torch.jit.trace(torch_model, torch.randn(input_shape)).eval()
    mod, params = tvm.relay.frontend.from_pytorch(scripted_model, [(input_name, input_shape)])
    return IRModPack(mod, params)

def get_irmod(
    model_name, dataset, mode, batch_size, image_size, ep_path=None, nchannels=3,
    zero_extra_params=False, include_extra_params=True, allow_ep_load_failure=False,
):
    """Loads a stored model and its params (including extra params for the
    specified mode) from the disk.
    zero_extra_params: If true, zero out the extra parameters in the model.
    Used for range building."""

    input_name = "input0"
    input_shape = (batch_size, nchannels, image_size, image_size)
    if model_name.startswith('Q'):
        fname = f'{data_root}/models/{dataset}/{model_name}/{model_name}-{batch_size}.onnx'
        mod, params = tvm.relay.frontend.from_onnx(onnx.load(fname), {'input0': input_shape})
    else:
        torch_model = get_torch_mod(model_name, dataset)
        mod, params = torch2irmod(torch_model, input_shape, input_name)

    if mode not in ['none', 'rb', 'cb', 'npcb', 'gn1', 'gn2', 'gninf', 'odin'] and include_extra_params:
        ep_load_fn = load_covs
        if mode == 'WNBC':
            mode = 'NBC'
        elif mode == 'npcv':
            mode = 'npc'
        if mode in {'NBC', 'KMN'}:
            ep_load_fn = load_bounds
        try:
            extra_params = ep_load_fn(model_name, dataset, mode, data_path=ep_path)
            extra_params_dict = {f'__ep_{idx}': v for idx, v in enumerate(extra_params)}
            params = {**params, **extra_params_dict}
        except Exception as e:
            if not allow_ep_load_failure:
                raise e
            utils.warn(f'Will skip loading extra params because of error: {e}')

    if zero_extra_params:
        params = extra_params_zeroed(params)

    return IRModPack(mod, params)

def load_module(path, debug=False, debug_dump_root=None):
    libmod = tvm.runtime.load_module(path)
    return create_rtmod(libmod, debug=debug, debug_dump_root=debug_dump_root)

def build_module(irmod, params, tunelog=None, export_path=None):
    start_time = time.time()
    with tvm.transform.PassContext(opt_level=3):
        if tunelog:
            with tvm.autotvm.apply_history_best(tunelog):
                lib = tvm.relay.build(irmod, target=target, params=params)
        else:
            lib = tvm.relay.build(irmod, target=target, params=params)
    rtmod = create_rtmod(lib, debug=False)
    if export_path:
        dirpath = os.path.dirname(export_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(os.path.dirname(export_path))
        lib.export_library(f'{export_path}')
    print(f'Module built in {time.time() - start_time:2f} seconds.')
    if export_path:
        print(f'Saved to {export_path}.')
    return rtmod, lib

def get_llvm_ir(irmod, params):
    _graph_json, rtmod, _params = tvm.relay.BuildModule().build(irmod, target=target, params=params)
    return rtmod.get_source()

def run_module(rtmod, input_data, output_defs=None, output_bufs=None, input_name='input0',
        default_nclasses=10, output_labels=False,
        benchmark=False, benchmark_std_threshold=1e-2, benchmark_max_trials=10):
    """Note that debug timing may not work when benchmark is enabled."""

    rtmod.set_input(input_name, input_data)
    exec_time, perf = None, None
    if benchmark:
        start_time = time.time()
        dev = tvm.device('cpu')
        for _ in range(benchmark_max_trials):
            perf = rtmod.benchmark(dev, number=100, repeat=3)
            if perf.std < perf.mean * benchmark_std_threshold:
                break
        else:
            utils.warn(f'Benchmark did not achieve desired stddev.')
        exec_time = time.time() - start_time
    else:
        rtmod.run()

    if not output_defs:
        output_defs = [{'shape': [input_data.shape[0], default_nclasses], 'dtype': 'float32'}]

    if not output_bufs:
        output_bufs = make_output_bufs(output_defs)

    outputs = [rtmod.get_output(i, buf).numpy() for i, buf in enumerate(output_bufs)]

    if output_labels:
        scores = softmax(outputs[0], axis=1)
        labels = np.argsort(scores, axis=1)[:, ::-1]
        outputs[0] = labels

    return ModRunResult(outputs, exec_time, perf)

def run_module_over_loader(rtmod, data_loader, outfile=None, **kwargs):
    """Runs the module over the data loader.
    Returns a list of ModRunResult objects."""
    ret = []
    if outfile and '/' in outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    for i, (xs, ys) in enumerate(tqdm(data_loader)):
        ret.append(run_module(rtmod, xs, **kwargs))
        if i % 100 == 0 and outfile:
            torch.save(ret, outfile)
    if outfile:
        print(f'Saved to {outfile}.')
    return ret
